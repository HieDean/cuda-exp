#include "transpose_vx.h"

__global__ void transpose_kernel_v0(const float *A, float *B, int m, int n)
{
    // int x0 = blockIdx.x * blockDim.x;
    // int xx = threadIdx.x;
    int xx = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;

    if (yy < m && xx < n) {
        B[xx * m + yy] = A[yy * n + xx];
    }

    return;
}

int transpose_v0(const float *A, float *B, int m, int n, cudaStream_t stream)
{
    /**
     * 为了缓解写入 B 矩阵时由于地址不连续导致的访存不合并现象, 应该尽可能的减小 blockDimX
     * x 方向上, 8 个 float 为 32 字节, 在内存对齐且地址连续的情况下可以合并为一次 L2 内存事务
     * 因此 blockDimX 设置为 8, 再小的话, 就会造成 sector 的浪费
     * 事实上 blockDimX 设置为 8 或者 4 都是可以的, 参考: https://zhuanlan.zhihu.com/p/1899760505733756129
     * 当 blockDimX = 8 时, warpDimX = 8, warpDimY = 4 所以读取 A 矩阵会产生 4 个 memory transactions
     * 同时写入 B 矩阵会产生 8 个 memory transactions
     * 当 blockDimX = 4 时, warpDimX = 4, warpDimY = 8 所以读取 A 矩阵会产生 8 个 memory transactions
     * 同时写入 B 矩阵会产生 4 个 memory transactions
     * 当 blockDimX 为其他值时, 读写产生的 memory transactions 数量会更多
     */
    constexpr int blockDimX = 8;
    constexpr int blockDimY = 32;
    dim3 blockDims(blockDimX, blockDimY);

    int gridDimX = (n + blockDimX - 1) / blockDimX;
    int gridDimY = (m + blockDimY - 1) / blockDimY;
    dim3 gridDims(gridDimX, gridDimY);

    transpose_kernel_v0<<<gridDims, blockDims, 0, stream>>>(A, B, m, n);
    return 0;
}