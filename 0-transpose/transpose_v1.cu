#include "transpose_vx.h"

template <int TILE_Y, int TILE_X, int WORKLOAD_Y, int WORKLOAD_X>
__global__ void transpose_kernel_v1(const float *A, float *B, int m, int n)
{
    int y0 = blockIdx.y * TILE_Y;
    int x0 = blockIdx.x * TILE_X;
    
    for (int rr = 0; rr < WORKLOAD_Y; ++rr) {
        int yy = y0 + rr * blockDim.y + threadIdx.y;
        for (int cc = 0; cc < WORKLOAD_X; ++cc) {
            int xx = x0 + cc * blockDim.x + threadIdx.x;
            if (yy < m && xx < n) {
                B[xx * m + yy] = A[yy * n + xx];
            }
        }
    }

    return;
}

int transpose_v1(const float *A, float *B, int m, int n, cudaStream_t stream)
{
    /**
     * v0 的问题:
     * 1. 写入 B 矩阵的时候, 一个 warp 访问的全局内存是不连续的, 这会产生多个内存事务;
     * 2. 一个线程只负责一个 float 线程的负载太低;
     */
    constexpr int blockDimY = 32;
    constexpr int blockDimX = 8;
    dim3 blockDims(blockDimX, blockDimY);

    constexpr int workloadY = 4;
    constexpr int workloadX = 1;
    constexpr int tileY = workloadY * blockDimY;
    constexpr int tileX = workloadX * blockDimX;

    int gridDimY = (m + tileY - 1) / tileY;
    int gridDimX = (n + tileX - 1) / tileX;
    dim3 gridDims(gridDimX, gridDimY);

    transpose_kernel_v1<tileY, tileX, workloadY, workloadX>
        <<<gridDims, blockDims, 0, stream>>>(A, B, m, n);
    return 0;
}