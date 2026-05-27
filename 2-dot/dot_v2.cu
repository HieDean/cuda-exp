#include "dot_vx.h"

template <int BLOCKSIZE>
__global__ void dot_kernel_v2_2(const float *tmp, float *C, int length, int load)
{
    int xx = threadIdx.x;
    __shared__ float smem[BLOCKSIZE];

    for (int ii = 0; ii < load; ++ii)
    {
        int xxx = ii * BLOCKSIZE + xx;
        smem[xx] += xxx < length ? tmp[xxx] : 0.0f;
    }

    // reduce
    for (int stride = BLOCKSIZE >> 1; stride > 32; stride >>= 1)
    {
        if (xx < stride)
        {
            smem[xx] += smem[xx + stride];
        }
        __syncthreads();
    }

    if (xx < 32)
    {
        smem[xx] = smem[xx] + smem[xx + 32];
        __syncwarp();
        smem[xx] = smem[xx] + smem[xx + 16];
        __syncwarp();
        smem[xx] = smem[xx] + smem[xx + 8];
        __syncwarp();
        smem[xx] = smem[xx] + smem[xx + 4];
        __syncwarp();
        smem[xx] = smem[xx] + smem[xx + 2];
        __syncwarp();
        smem[xx] = smem[xx] + smem[xx + 1];
    }

    if (xx == 0)
    {
        C[0] = smem[0];
    }
}

template <int TILESIZE, int TILEFACTOR, int BLOCKSIZE>
__global__ void dot_kernel_v2_1(const float *A, const float *B, float *tmp, int length)
{
    int x0 = blockIdx.x * blockDim.x;
    int xx = threadIdx.x;

    __shared__ float smem[BLOCKSIZE];

    smem[xx] = 0.0f;
    for (int ii = 0; ii < TILEFACTOR; ++ii)
    {
        int xxx = x0 + ii * TILESIZE + xx;
        smem[xx] += xxx < length ? A[xxx] * B[xxx] : 0.0f;
    }

    __syncthreads();

    // reduce
    for (int stride = BLOCKSIZE >> 1; stride > 32; stride >>= 1)
    {
        if (xx < stride)
        {
            smem[xx] += smem[xx + stride];
        }
        __syncthreads();
    }

    // 当 stride 小于 32 时, 手动进行循环展开并只进行 warp 内同步会更快
    if (xx < 32)
    {
        smem[xx] = smem[xx] + smem[xx + 32];
        __syncwarp();
        smem[xx] = smem[xx] + smem[xx + 16];
        __syncwarp();
        smem[xx] = smem[xx] + smem[xx + 8];
        __syncwarp();
        smem[xx] = smem[xx] + smem[xx + 4];
        __syncwarp();
        smem[xx] = smem[xx] + smem[xx + 2];
        __syncwarp();
        smem[xx] = smem[xx] + smem[xx + 1];
    }

    if (xx == 0)
    {
        tmp[blockIdx.x] = smem[0];
    }

    return;
}

int dot_v2(const float *A, const float *B, float *C, int length, cudaStream_t stream)
{
    // 图一乐, v1 中仍然有 numBlocksPerGrid 个线程竞争同一个原子变量, 这让我不禁手痒, 能不能不用原子操作?
    // 不用原子操作会引入额外的全局内存, 不一定会更快
    // 在访问 A B 矩阵时没有考虑使用 FLOAT4, 目的是不舍弃 kernel 的通用性
    constexpr int numThreadsPerBlock = 128;
    constexpr int tileFactor = 4;
    constexpr int tileSize = numThreadsPerBlock * tileFactor;
    int numBlocksPerGrid = (length + tileSize - 1) / tileSize;

    dim3 blockDims(numThreadsPerBlock);
    dim3 gridDims(numBlocksPerGrid);

    float *tmp;
    cudaMalloc(reinterpret_cast<void **>(&tmp), numBlocksPerGrid * sizeof(float));

    dot_kernel_v2_1<tileSize, tileFactor, numThreadsPerBlock>
        <<<gridDims, blockDims, 0, stream>>>(A, B, tmp, length);

    constexpr int numThreads = 1024;
    int load = (numBlocksPerGrid + numThreads - 1) / numThreads;
    dot_kernel_v2_2<numThreads>
        <<<1, numThreads, 0, stream>>>(tmp, C, numBlocksPerGrid, load);

    cudaFree(tmp);
    return 0;
}