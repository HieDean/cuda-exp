#include "gemm_vx.h"

template <int TILE_M, int TILE_N, int TILE_K, int STRIDE>
__global__ void gemm_kernel_v2(const float *A, const float *B, float *C,
                               int m, int n, int k)
{
    int tile_start_y = blockIdx.y * TILE_M;
    int tile_start_x = blockIdx.x * TILE_N;
    int bk = TILE_K / STRIDE;

    // 是否超越了 C 的边缘
    if (tile_start_y >= m || tile_start_x >= n)
    {
        return;
    }

    extern __shared__ char smem[];
    float *smem_a = (float *)smem;            // [TILE_M, TILE_K]
    float *smem_b = &smem_a[TILE_M * TILE_K]; // [TILE_K, TILE_N]

    float sum[STRIDE * STRIDE] = {};
    for (int ss = 0; ss < k; ss += TILE_K)
    {
        for (int rr = 0; rr < STRIDE; ++rr)
        {
            for (int cc = 0; cc < STRIDE; ++cc)
            {
                // 我们需要使用线程id来对a b smem进行索引
                /**smem_a
                 * y: rr * blockDim.y + threadIdx.y
                 * x: cc * bk + threadIdx.x
                 * side_length: TILE_K
                 * 当 blockDim.x == blockDim.y == bk 的时候, smem_a和smem_b的索引可以复用
                 * smem_b
                 * y: rr * bk + threadIdx.y
                 * x: cc * blockDim.x + threadIdx.x
                 * side_length: blockDim.x * STRIDE
                 */
                /**device_a m*k
                 * y: tile_start_y + rr * blockDim.y + threadIdx.y
                 * x: ss + cc * bk + threadIdx.x
                 */
                /**device_b k*n
                 * y: ss + rr * bk + threadIdx.y
                 * x: tile_start_x + cc * blockDim.x + threadIdx.x
                 */
                int idsa_y = rr * blockDim.y + threadIdx.y;
                int idsa_x = cc * bk + threadIdx.x;
                int ida_y = tile_start_y + idsa_y;
                int ida_x = ss + idsa_x;
                if (ida_y >= m || ida_x >= k)
                {
                    smem_a[idsa_y * TILE_K + idsa_x] = 0.0;
                }
                else
                {
                    smem_a[idsa_y * TILE_K + idsa_x] = A[ida_y * k + ida_x];
                }

                int idsb_y = rr * bk + threadIdx.y;
                int idsb_x = cc * blockDim.x + threadIdx.x;
                int idb_y = ss + idsb_y;
                int idb_x = tile_start_x + idsb_x;
                if (ss + rr * bk + threadIdx.y >= k || tile_start_x + cc * blockDim.x + threadIdx.x >= n)
                {
                    smem_b[idsb_y * TILE_N + idsb_x] = 0.0;
                }
                else
                {
                    smem_b[idsb_y * TILE_N + idsb_x] = B[idb_y * n + idb_x];
                }
            }
        }
        __syncthreads();

        for (int rr = 0; rr < STRIDE; ++rr)
        {
            for (int cc = 0; cc < STRIDE; ++cc)
            {
                // calc the [rr, cc] tile of c, each tile will resolved by one block.
                for (int ii = 0; ii < TILE_K; ++ii)
                {
                    /**smem_a
                     * y: rr * blockDim.y + threadIdx.y
                     * x: ii
                     * side_length: TILE_K
                     */
                    /**smem_b
                     * y: ii
                     * x: cc * blockDim.x + threadIdx.x
                     * side_length: blockDim.x * STRIDE
                     */
                    sum[rr * STRIDE + cc] +=
                        smem_a[(rr * blockDim.y + threadIdx.y) * TILE_K + ii] *
                        smem_b[ii * TILE_N + cc * blockDim.x + threadIdx.x];
                }
            }
        }
        __syncthreads();
    }

    for (int rr = 0; rr < STRIDE; ++rr)
    {
        for (int cc = 0; cc < STRIDE; ++cc)
        {
            /**
             * y: tile_start_y + rr * blockDim.y + threadIdx.y
             * x: tile_start_x + cc * blockDim.x + threadIdx.x
             * side_length: n
             */
            if (tile_start_y + rr * blockDim.y + threadIdx.y < m &&
                tile_start_x + cc * blockDim.x + threadIdx.x < n)
            {
                C[(tile_start_y + rr * blockDim.y + threadIdx.y) * n +
                  tile_start_x + cc * blockDim.x + threadIdx.x] = sum[rr * STRIDE + cc];
            }
        }
    }

    return;
}

int gemm_v2(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream)
{
    // 使用 32 * 32 个线程, 计算 C 中的 64 * 64 个点
    const int tile_m = 64;
    const int tile_n = 64;
    const int tile_k = 64;
    const int stride = 2;

    const int bm = tile_m / stride;
    const int bn = tile_n / stride;
    dim3 blockDims(bn, bm);

    int gridDim_x = (n + tile_n - 1) / (tile_n);
    int gridDim_y = (m + tile_m - 1) / (tile_m);
    dim3 gridDims(gridDim_x, gridDim_y);

    int smemSize = tile_m * tile_k + tile_n * tile_k;
    gemm_kernel_v2<tile_m, tile_n, tile_k, stride>
        <<<gridDims, blockDims, smemSize * sizeof(float), stream>>>(A, B, C, m, n, k);
    return 0;
}