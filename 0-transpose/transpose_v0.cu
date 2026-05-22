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
     * 因此 blockDimX 设置为8, 再小的话, 就会造成 sector 的浪费
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