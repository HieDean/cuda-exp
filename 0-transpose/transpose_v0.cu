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
    constexpr int blockDimX = 16;
    constexpr int blockDimY = 16;
    dim3 blockDims(blockDimX, blockDimY);

    int gridDimX = (n + blockDimX - 1) / blockDimX;
    int gridDimY = (m + blockDimY - 1) / blockDimY;
    dim3 gridDims(gridDimX, gridDimY);

    transpose_kernel_v0<<<gridDims, blockDims, 0, stream>>>(A, B, m, n);
    return 0;
}