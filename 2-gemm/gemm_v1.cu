#include "gemm_vx.h"

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_kernel_v1(const float *A, const float *B, float *C,
                               int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n)
    {
        return;
    }

    extern __shared__ char smem_v1[];
    float *smem_a = (float *)smem_v1;             // [blockDim.y == TILE_M, TILE_K]
    float *smem_b = &smem_a[blockDim.y * TILE_K]; // [TILE_K, blockDim.x == TILE_N]

    float sum = 0.0;
    for (int ss = 0; ss < k; ss += TILE_K)
    {
        /**
         * 访问 global memory 中的 A 矩阵的时候, block 内部 x 方向的线程访问连续的地址, 所以 x 方向的访存可以合并(coalesced);
         * 访问 global memory 中的 B 矩阵的时候, block 内部 x 方向的线程访问连续的地址, 所以 x 方向的访存也可以合并(coalesced);
         * ----------------------------------------------------
         * 这里写入 shared memory 的时候, 当 threadIdx.y 固定, threadIdx.x 从 0 到 31 访问 smem_a 和 smem_b 的时候,
         * 访问的是长度为 32 的连续地址, 恰好访问 32 个不同的 bank, 不存在 bank conflict;
         */

        int ida_y = row;
        int ida_x = ss + threadIdx.x;
        if (ida_y >= m || ida_x >= n)
        {
            smem_a[threadIdx.y * TILE_K + threadIdx.x] = 0.0;
        }
        else
        {
            smem_a[threadIdx.y * TILE_K + threadIdx.x] = A[ida_y * k + ida_x];
        }

        int idb_y = ss + threadIdx.y;
        int idb_x = col;
        if (idb_y >= m || idb_x >= n)
        {
            smem_b[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
        }
        else
        {
            smem_b[threadIdx.y * blockDim.x + threadIdx.x] = B[(ss + threadIdx.y) * n + col];
        }

        __syncthreads();

        /**
         * 这里访问 shared memory:
         * 当 threadIdx.y 固定, threadIdx.x 从 0 到 31 访问 smem_a 的时候, 访问的是相同的地址, 即 smem_a[threadIdx.y * TILE_K + ii], 因此触发 broadcast;
         * 当 threadIdx.y 固定, threadIdx.x 从 0 到 31 访问 smem_b 的时候, 访问的是长度为 32 的连续地址, 恰好访问 32 个不同的 bank, 不存在 bank conflict;
         * PS: bank 布局: smem[0](bank0) smem[1](bank1) smem[2](bank2) ... smem[31](bank31) smem[32](bank0) ...
         */
        for (int ii = 0; ii < TILE_K; ++ii)
        {
            sum += smem_a[threadIdx.y * TILE_K + ii] * smem_b[ii * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }

    C[row * n + col] = sum;

    return;
}

int gemm_v1(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream)
{
    const int tile_m = 32;
    const int tile_n = 32;
    const int tile_k = 32;

    const int blockDim_x = 32;
    const int blockDim_y = 32;
    dim3 blockDims(blockDim_x, blockDim_y);

    int gridDim_x = (n + blockDim_x - 1) / blockDim_x;
    int gridDim_y = (m + blockDim_y - 1) / blockDim_y;
    dim3 gridDims(gridDim_x, gridDim_y);

    int smemSize = blockDim_x * tile_k + blockDim_y * tile_k;
    gemm_kernel_v1<tile_m, tile_n, tile_k>
        <<<gridDims, blockDims, smemSize * sizeof(float), stream>>>(A, B, C, m, n, k);
    return 0;
}