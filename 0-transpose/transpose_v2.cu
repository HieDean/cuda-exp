#include "transpose_vx.h"

template <int BLOCKDIM_AY, int BLOCKDIM_AX, int BLOCKDIM_BY, int BLOCKDIM_BX,
          int WORKLOAD_AY, int WORKLOAD_AX, int WORKLOAD_BY, int WORKLOAD_BX,
          int TILE_Y, int TILE_X>
__global__ void transpose_kernel_v2(const float *A, float *B, int m, int n)
{
    __shared__ float smem[TILE_X][TILE_Y];

    // 在 A 中是 tileY * tileX
    int r0 = blockIdx.y * TILE_Y;
    int c0 = blockIdx.x * TILE_X;

    int tIdr = threadIdx.x / BLOCKDIM_AX;
    int tIdc = threadIdx.x % BLOCKDIM_AX;

    // 在 B 中是 tileX * tileY
    int _r0 = blockIdx.x * TILE_X;
    int _c0 = blockIdx.y * TILE_Y;

    int _tIdr = threadIdx.x / BLOCKDIM_BX;
    int _tIdc = threadIdx.x % BLOCKDIM_BX;

    for (int rr = 0; rr < WORKLOAD_AY; ++rr)
    {
        int srow = rr * BLOCKDIM_AY + tIdr;
        int row = r0 + srow;
        for (int cc = 0; cc < WORKLOAD_AX; ++cc)
        {
            int scol = cc * BLOCKDIM_AX + tIdc;
            int col = c0 + scol;
            if (row < m && col < n)
            {
                /**
                 * 从 A 矩阵读入数据并直接转置写入共享内存
                 * 这里没有 swizzle 会存在 bank conflict
                 */
                smem[scol][(srow + scol) & (TILE_Y - 1)] = A[row * n + col];
            }
        }
    }

    __syncthreads();

    for (int rr = 0; rr < WORKLOAD_BY; ++rr)
    {
        int srow = rr * BLOCKDIM_BY + _tIdr;
        int row = _r0 + srow;
        for (int cc = 0; cc < WORKLOAD_BX; ++cc)
        {
            int scol = cc * BLOCKDIM_BX + _tIdc;
            int col = _c0 + scol;
            if (row < n && col < m)
            {
                /**
                 * 从共享内存读入数据并写入 B 矩阵
                 * 这里没有 swizzle 会存在 bank conflict
                 */
                B[row * m + col] = smem[srow][(srow + scol) & (TILE_Y - 1)];
            }
        }
    }

    return;
}

int transpose_v2(const float *A, float *B, int m, int n, cudaStream_t stream)
{
    /**
     * v1 解决了线程负载低的问题, 但写入 B 矩阵的时候, 仍然存在访存不合并现象;
     * v2 使用共享内存, 在共享内存中进行转置, 然后以访存完全合并的方式写入 B 矩阵;
     */
    constexpr int blockDimAy = 32;
    constexpr int blockDimAx = 8;
    constexpr int blockDimBy = 8;
    constexpr int blockDimBx = 32;
    // assert(blockDimAy * blockDimAx == blockDimBy * blockDimBx);
    dim3 blockDims(blockDimAx * blockDimAy);

    constexpr int workloadAy = 4;
    constexpr int workloadAx = 1;
    constexpr int workloadBy = 1;
    constexpr int workloadBx = 4;
    // assert(workloadAy * workloadAx == workloadBy * workloadBx);

    // 在 A 中是 tileY * tileX, 在 B 中是 tileX * tileY
    constexpr int tileY = workloadAy * blockDimAy; // 4 * 32
    constexpr int tileX = workloadAx * blockDimAx; // 1 * 8
    int gridDimY = (m + tileY - 1) / tileY;
    int gridDimX = (n + tileX - 1) / tileX;
    dim3 gridDims(gridDimX, gridDimY);

    transpose_kernel_v2<blockDimAy, blockDimAx, blockDimBy, blockDimBx,
                        workloadAy, workloadAx, workloadBy, workloadBx,
                        tileY, tileX>
        <<<gridDims, blockDims, 0, stream>>>(A, B, m, n);
    return 0;
}