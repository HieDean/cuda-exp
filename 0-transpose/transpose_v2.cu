#include "transpose_vx.h"

template <int BLOCKDIM_Y, int BLOCKDIM_X, int TILE_Y, int TILE_X, int WORKLOAD_Y, int WORKLOAD_X>
__global__ void transpose_kernel_v2(const float *A, float *B, int m, int n)
{
    __shared__ float smem[TILE_X][TILE_Y];

    int r0 = blockIdx.y * TILE_Y;
    int c0 = blockIdx.x * TILE_X;

    int tIdr = threadIdx.x / BLOCKDIM_X;
    int tIdc = threadIdx.x % BLOCKDIM_X;

    int _r0 = blockIdx.x * TILE_X;
    int _c0 = blockIdx.y * TILE_Y;

    int _tIdr = threadIdx.x / BLOCKDIM_Y;
    int _tIdc = threadIdx.x % BLOCKDIM_Y;

    for (int rr = 0; rr < WORKLOAD_Y; ++rr)
    {
        int srow = rr * BLOCKDIM_Y + tIdr;
        int row = r0 + srow;
        for (int cc = 0; cc < WORKLOAD_X; ++cc)
        {
            int scol = cc * BLOCKDIM_X + tIdc;
            int col = c0 + scol;
            if (row < m && col < n)
            {
                /**
                 * 从 A 矩阵读入数据并直接转置写入共享内存
                 * 这里没有 swizzle, 会存在 bank conflict
                 */
                smem[scol][srow] = A[row * n + col];
            }
        }
    }

    __syncthreads();

    for (int rr = 0; rr < WORKLOAD_X; ++rr)
    {
        int srow = rr * BLOCKDIM_X + _tIdr;
        int row = _r0 + srow;
        for (int cc = 0; cc < WORKLOAD_Y; ++cc)
        {
            int scol = cc * BLOCKDIM_Y + _tIdc;
            int col = _c0 + scol;
            if (row < n && col < m)
            {
                /**
                 * 从共享内存读入数据并写入 B 矩阵
                 * 这里没有 swizzle, 会存在 bank conflict
                 */
                B[row * m + col] = smem[srow][scol];
            }
        }
    }

    return;
}

int transpose_v2(const float *A, float *B, int m, int n, cudaStream_t stream)
{
    /**
     * v1 的问题:
     * 1. 虽然一个线程可以处理多个元素, 但在写入 B 矩阵的时候, 仍然可以在访存合并方便进行优化;
     */
    constexpr int blockDimY = 32;
    constexpr int blockDimX = 8;
    dim3 blockDims(blockDimX * blockDimY);

    constexpr int workloadY = 4;
    constexpr int workloadX = 1;
    constexpr int tileY = workloadY * blockDimY;
    constexpr int tileX = workloadX * blockDimX;

    int gridDimY = (m + tileY - 1) / tileY;
    int gridDimX = (n + tileX - 1) / tileX;
    dim3 gridDims(gridDimX, gridDimY);

    transpose_kernel_v2<blockDimY, blockDimX, tileY, tileX, workloadY, workloadX>
        <<<gridDims, blockDims, 0, stream>>>(A, B, m, n);
    return 0;
}