#include "gemm_vx.h"

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template <int TILE_M, int TILE_N, int TILE_K, int BLOCKDIM_X_A, int BLOCKDIM_X_B, int BLOCKDIM_X_C, int WARPDIM_X, int BLOCKDIM>
__global__ void gemm_kernel_v7(const float *A, const float *B, float *C,
                               int m, int n, int k)
{
    int tile_y0 = blockIdx.y * TILE_M;
    int tile_x0 = blockIdx.x * TILE_N;

    if (tile_y0 >= m || tile_x0 >= n)
    {
        return;
    }

    // 计算 y 方向大小
    constexpr int BLOCKDIM_Y_A = BLOCKDIM / BLOCKDIM_X_A;
    constexpr int BLOCKDIM_Y_B = BLOCKDIM / BLOCKDIM_X_B;
    constexpr int BLOCKDIM_Y_C = BLOCKDIM / BLOCKDIM_X_C;

    // 每个线程负责计算 C 中的 WORKLOAD_Y*WORKLOAD_X 个数
    constexpr int WORKLOAD_Y = TILE_M / BLOCKDIM_Y_C;
    constexpr int WORKLOAD_X = TILE_N / BLOCKDIM_X_C;
    float workload[WORKLOAD_Y * WORKLOAD_X] = {};

    // 计算线程在不同组织情况下, x 和 y 向的偏移
    int tid = threadIdx.x;
    int tid_y_a = tid / BLOCKDIM_X_A;
    int tid_x_a = tid % BLOCKDIM_X_A;
    int tid_y_b = tid / BLOCKDIM_X_B;
    int tid_x_b = tid % BLOCKDIM_X_B;
    // int tid_y_c = tid / BLOCKDIM_X_C;
    // int tid_x_c = tid % BLOCKDIM_X_C;

    constexpr int WARP_SIZE = 32;
    constexpr int WARPDIM_Y = WARP_SIZE / WARPDIM_X;
    // constexpr int NUM_WARPS = BLOCKDIM / WARP_SIZE;
    constexpr int NUM_WARPS_X = BLOCKDIM_X_C / WARPDIM_X;
    // constexpr int NUM_WARPS_Y = BLOCKDIM_Y_C / WARPDIM_Y;
    int warp_id = tid / WARP_SIZE;
    int warp_id_y = warp_id / NUM_WARPS_X;
    int warp_id_x = warp_id % NUM_WARPS_X;
    int lane_id = tid % WARP_SIZE;
    int lane_id_y = lane_id % 2 + lane_id / 16 * 2; // z-order
    int lane_id_x = lane_id % 16 / 2;               // z-order
    int rewarp_id_y = warp_id_y * WARPDIM_Y + lane_id_y;
    int rewarp_id_x = warp_id_x * WARPDIM_X + lane_id_x;

    extern __shared__ char smem[];
    float *smem_a = (float *)smem;                // [2, TILE_K, TILE_M] // transpose
    float *smem_b = &smem_a[2 * TILE_K * TILE_M]; // [2, TILE_K, TILE_N]
    constexpr int BUFDIM_A = TILE_K * TILE_M;
    constexpr int BUFDIM_B = TILE_K * TILE_N;

    float smem_a_xx_kk[2][WORKLOAD_Y];
    float smem_b_kk_xx[2][WORKLOAD_X];

    int buf_id = 0;

#pragma unroll
    for (int ks = 0; ks < k + TILE_K; ks += TILE_K)
    {
        if (ks > 0)
        {
// 对 smem_a 和 smem_b 做矩阵乘
#pragma unroll
            for (int kk = 0; kk < TILE_K + 1; ++kk)
            {
                if (kk > 0)
                {
#pragma unroll
                    for (int idw_y = 0; idw_y < WORKLOAD_Y; ++idw_y)
                    {
#pragma unroll
                        for (int idw_x = 0; idw_x < WORKLOAD_X; ++idw_x)
                        {
                            workload[idw_y * WORKLOAD_X + idw_x] += smem_a_xx_kk[(kk - 1) % 2][idw_y] *
                                                                    smem_b_kk_xx[(kk - 1) % 2][idw_x];
                        }
                    }
                }

                if (kk < TILE_K)
                {
#pragma unroll
                    for (int idw_y = 0; idw_y < WORKLOAD_Y; idw_y += 4)
                    {
                        int swizzle = (BLOCKDIM_Y_C * idw_y + rewarp_id_y * 4) ^ (4 * kk);
                        FLOAT4(smem_a_xx_kk[kk % 2][idw_y]) =
                            FLOAT4(smem_a[(buf_id ^ 1) * BUFDIM_A + kk * TILE_M + swizzle]);
                    }
#pragma unroll
                    for (int idw_x = 0; idw_x < WORKLOAD_X; idw_x += 4)
                    {
                        int idb_x = BLOCKDIM_X_C * idw_x + rewarp_id_x * 4;
                        FLOAT4(smem_b_kk_xx[kk % 2][idw_x]) =
                            FLOAT4(smem_b[(buf_id ^ 1) * BUFDIM_B + kk * TILE_N + idb_x]);
                    }
                }
            }

            __syncthreads();
        }

        if (ks < k)
        {
// 使用 BLOCKDIM_Y_A*BLOCKDIM_X_A 个线程加载 TILE_M*TILE_K 个数到 smem_a
#pragma unroll
            for (int rs = 0; rs < TILE_M; rs += BLOCKDIM_Y_A)
            {
                int idsa_y = rs + tid_y_a;
                int ida_y = tile_y0 + idsa_y;
#pragma unroll
                for (int cs = 0; cs < TILE_K; cs += BLOCKDIM_X_A)
                {
                    int idsa_x = cs + tid_x_a;
                    int ida_x = ks + idsa_x;
                    int swizzle = idsa_y ^ (4 * idsa_x); // swizzle
                    smem_a[buf_id * BUFDIM_A + idsa_x * TILE_M + swizzle] =
                        (ida_y < m && ida_x < k) ? A[ida_y * k + ida_x] : 0.0f; // transpose
                }
            }

// 使用 BLOCKDIM_Y_B*BLOCKDIM_X_B 个线程加载 TILE_K*TILE_N 个数到 smem_b
#pragma unroll
            for (int rs = 0; rs < TILE_K; rs += BLOCKDIM_Y_B)
            {
                int idsb_y = rs + tid_y_b;
                int idb_y = ks + idsb_y;
#pragma unroll
                for (int cs = 0; cs < TILE_N; cs += BLOCKDIM_X_B)
                {
                    int idsb_x = cs + tid_x_b;
                    int idb_x = tile_x0 + idsb_x;
                    smem_b[buf_id * BUFDIM_B + idsb_y * TILE_N + idsb_x] =
                        (idb_y < k && idb_x < n) ? B[idb_y * n + idb_x] : 0.0f;
                }
            }

            __syncthreads();
        }
        buf_id ^= 1;
    }

// 使用 BLOCKDIM_Y_C*BLOCKDIM_X_C 个线程将 BLOCKDIM_Y_C*BLOCKDIM_X_C 个数写入 C
// 每个线程写入 WORKLOAD_Y * WORKLOAD_X 个数
#pragma unroll
    for (int idw_y = 0; idw_y < WORKLOAD_Y; ++idw_y)
    {
        int idc_y = tile_y0 + (rewarp_id_y << 2) + ((idw_y >> 2) << 2) * BLOCKDIM_Y_C + (idw_y & 3);
#pragma unroll
        for (int idw_x = 0; idw_x < WORKLOAD_X; ++idw_x)
        {
            int idc_x = tile_x0 + (rewarp_id_x << 2) + ((idw_x >> 2) << 2) * BLOCKDIM_X_C + (idw_x & 3);
            if (idc_y < m && idc_x < n)
            {
                C[idc_y * n + idc_x] = workload[idw_y * WORKLOAD_X + idw_x];
            }
        }
    }

    return;
}

int gemm_v7(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream)
{
    constexpr int tile_m = 128;
    constexpr int tile_n = 128;
    constexpr int tile_k = 8;
    constexpr int blockDim_x_a = 8;
    constexpr int blockDim_x_b = 32;
    constexpr int blockDim_x_c = 16;
    constexpr int warpDim_x = 8;
    constexpr int numThreadsPerBlock = 256;
    dim3 blockDims(numThreadsPerBlock);

    int gridDim_x = (n + tile_n - 1) / (tile_n);
    int gridDim_y = (m + tile_m - 1) / (tile_m);
    dim3 gridDims(gridDim_x, gridDim_y);

    constexpr int smemSize = 2 * tile_m * tile_k + 2 * tile_n * tile_k;
    gemm_kernel_v7<tile_m, tile_n, tile_k, blockDim_x_a, blockDim_x_b, blockDim_x_c, warpDim_x, numThreadsPerBlock>
        <<<gridDims, blockDims, smemSize * sizeof(float), stream>>>(A, B, C, m, n, k);
    return 0;
}
