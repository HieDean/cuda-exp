#include "gemm_vx.h"

__global__ void gemm_kernel_v0(const float *A, const float *B, float *C,
                               int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n)
    {
        return;
    }

    float sum = 0.0;
    for (int kk = 0; kk < k; ++kk)
    {
        sum += A[row * k + kk] * B[kk * n + col];
    }
    C[row * n + col] = sum;

    return;
}

int gemm_v0(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream)
{
    const int blockDim_x = 32;
    const int blockDim_y = 32;
    dim3 blockDims(blockDim_x, blockDim_y);

    int gridDim_x = (n + blockDim_x - 1) / blockDim_x;
    int gridDim_y = (m + blockDim_y - 1) / blockDim_y;
    dim3 gridDims(gridDim_x, gridDim_y);

    gemm_kernel_v0<<<gridDims, blockDims, 0, stream>>>(A, B, C, m, n, k);
    return 0;
}