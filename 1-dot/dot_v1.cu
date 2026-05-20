#include "dot_vx.h"

template <int TILESIZE, int TILEFACTOR, int BLOCKSIZE>
__global__ void dot_kernel_v1(const float *A, const float *B, float *C, int length)
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
        atomicAdd(C, smem[0]);
    }

    return;
}

int dot_v1(const float *A, const float *B, float *C, int length, cudaStream_t stream)
{
    // 分析 v0, 有两个地方效率太低:
    // 1. 一个线程读两个数, 做一次乘法计算, 计算访存比太低;
    //    由于一次乘法计算必须读取两个数, 所以尽可能让访存合并, 通过少量访存, 大量计算的方式提高计算访存比;
    // 2. 最后的求和有多个线程竞争一个原子变量;
    //    采用 reduce 的方式, block 内部求和, 最后原子求和, 降低原子变量的竞争线程个数;

    // 一个 block 计算 tile 个数
    constexpr int numThreadsPerBlock = 128;
    constexpr int tileFactor = 4;
    constexpr int tileSize = numThreadsPerBlock * tileFactor;
    int numBlocksPerGrid = (length + tileSize - 1) / tileSize;

    dim3 blockDims(numThreadsPerBlock);
    dim3 gridDims(numBlocksPerGrid);

    dot_kernel_v1<tileSize, tileFactor, numThreadsPerBlock>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, length);
    return 0;
}