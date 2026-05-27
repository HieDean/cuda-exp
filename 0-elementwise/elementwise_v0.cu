#include "elementwise_vx.h"

__global__ void elementwise_kernel_v0(const float *A, const float *B, float *C, int m, int n)
{
    int xx = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;

    if (yy < m && xx < n) {
        C[yy * n + xx] = A[yy * n + xx] * B[yy * n + xx];
    }

    return;
}

int elementwise_v0(const float *A, const float *B, float *C, int m, int n, cudaStream_t stream)
{
    constexpr int blockDimX = 16;
    constexpr int blockDimY = 16;
    dim3 blockDims(blockDimX, blockDimY);

    int gridDimX = (n + blockDimX - 1) / blockDimX;
    int gridDimY = (m + blockDimY - 1) / blockDimY;
    dim3 gridDims(gridDimX, gridDimY);

    elementwise_kernel_v0<<<gridDims, blockDims, 0, stream>>>(A, B, C, m, n);
    return 0;
}