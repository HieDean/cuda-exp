#include "batchdot_vx.h"

template <int WARP_SIZE, int BLOCK_SIZE, int NUM_WARPS>
__global__ void batchdot_kernel_v0(const float *A, const float *B, float *C, int bs, int length)
{
    int x0 = blockIdx.x * length;
    int locx = threadIdx.x;
    int laneId = locx % WARP_SIZE;
    int warpId = locx / WARP_SIZE;

    __shared__ float shared_sum[NUM_WARPS];

    // loop sum in tile
    float sum = 0.0f;
    int numElements = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int ii = 0; ii < numElements; ++ii)
    {
        int index = x0 + ii * BLOCK_SIZE + locx;
        sum += ii * BLOCK_SIZE + locx < length ? A[index] * B[index] : 0.0f;
    }

    // reduce sum in warp
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1)
    {
        sum += __shfl_down_sync(0xffffffff, sum, mask, WARP_SIZE);
    }
    // broadcast
    if (laneId == 0)
    {
        shared_sum[warpId] = sum;
    }
    __syncthreads();

    // reduce sum again
    sum = locx < NUM_WARPS ? shared_sum[locx] : 0.0f;
    // for (int mask = NUM_WARPS >> 1; mask > 0; mask >>= 1)
    // {
    //     // NUM_WARPS = 256 / 32 = 8 => 0x000000ff
    //     sum += __shfl_down_sync(0x000000ff, sum, mask, NUM_WARPS);
    // } // 这种写法不通用
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1)
    {
        sum += __shfl_down_sync(0xffffffff, sum, mask, WARP_SIZE);
    }
    __syncthreads();

    if (locx == 0)
    {
        C[blockIdx.x] = sum;
    }

    return;
}

int batchdot_v0(const float *A, const float *B, float *C, int bs, int length, cudaStream_t stream)
{
    // 一个 block 计算 tile 个数
    constexpr int warpSize = 32;
    constexpr int numThreadsPerBlock = 256;
    constexpr int numWarpsPerBlock = numThreadsPerBlock / warpSize;
    int numBlocksPerGrid = bs;

    dim3 blockDims(numThreadsPerBlock);
    dim3 gridDims(numBlocksPerGrid);

    batchdot_kernel_v0<warpSize, numThreadsPerBlock, numWarpsPerBlock>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, length);
    return 0;
}