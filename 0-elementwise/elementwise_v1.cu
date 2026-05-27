#include "elementwise_vx.h"

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template <int BLOCKDIM_Y, int BLOCKDIM_X,
          int WORKLOAD_Y, int WORKLOAD_X,
          int TILE_Y, int TILE_X>
__global__ void elementwise_kernel_v1(const float *A, const float *B, float *C, int m, int n)
{
    int y0 = blockIdx.y * TILE_Y;
    int x0 = blockIdx.x * TILE_X;

    for (int rr = 0; rr < WORKLOAD_Y; ++rr)
    {
        int yy = y0 + rr * BLOCKDIM_Y + threadIdx.y;
        for (int cc = 0; cc < WORKLOAD_X; ++cc)
        {
            int xx = x0 + cc * BLOCKDIM_X + threadIdx.x;
            if (yy < m && xx < n) {
                C[yy * n + xx] = A[yy * n + xx] * B[yy * n + xx];
            }
        }
    }

    return;
}

int elementwise_v1(const float *A, const float *B, float *C, int m, int n, cudaStream_t stream)
{
    /**
     * v0 太 naive 了
     * v1 增加以下特性:
     * 1. 单个线程处理多个元素;
     * 2. 向量化访存, 但向量化访存会导致 kernel 失去通用性, 这里不会做实现;
     */
    constexpr int blockDimY = 16;
    constexpr int blockDimX = 16;
    dim3 blockDims(blockDimX, blockDimY);

    constexpr int workloadY = 2;
    constexpr int workloadX = 2;
    constexpr int tileY = blockDimY * workloadY;
    constexpr int tileX = blockDimX * workloadX;

    int gridDimY = (m + tileY - 1) / tileY;
    int gridDimX = (n + tileX - 1) / tileX;
    dim3 gridDims(gridDimX, gridDimY);

    elementwise_kernel_v1<blockDimY, blockDimX,
                          workloadY, workloadX,
                          tileY, tileX>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, m, n);
    return 0;
}