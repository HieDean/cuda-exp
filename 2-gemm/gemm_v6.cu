#include "gemm_vx.h"

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template <int TILE_M=128, int TILE_N=128, int TILE_K=8,
          int BLOCKDIM_X_A=8, int BLOCKDIM_Y_A=32,
          int BLOCKDIM_X_B=32, int BLOCKDIM_Y_B=8,
          int BLOCKDIM_X_C=16, int BLOCKDIM_Y_C=16,
          int WORKLOAD_X=8, int WORKLOAD_Y=8,
          int WARPDIM_X=8, int WARPDIM_Y=4,
          int NUM_WARP_X=2, int NUM_WARP_Y=4,
          int BLOCKDIM=256, int WARP_SIZE=32>
__global__ void gemm_kernel_v6(const float *A, const float *B, float *C,
                               int m, int n, int k)
{
    int tile_y0 = blockIdx.y * TILE_M;
    int tile_x0 = blockIdx.x * TILE_N;

    if (tile_y0 >= m || tile_x0 >= n)
    {
        return;
    }

    // 每个线程负责计算 C 中的 WORKLOAD_Y*WORKLOAD_X 个数
    float workload[WORKLOAD_Y][WORKLOAD_X] = {};

    // 计算线程在不同组织情况下, x 和 y 向的偏移
    int tid = threadIdx.x;
    int tid_y_a = tid / BLOCKDIM_X_A;
    int tid_x_a = tid & (BLOCKDIM_X_A - 1);
    int tid_y_b = tid / BLOCKDIM_X_B;
    int tid_x_b = tid & (BLOCKDIM_X_B - 1);

    int warp_id = tid / WARP_SIZE;
    int warp_id_y = warp_id / (NUM_WARP_X);
    int warp_id_x = warp_id & (NUM_WARP_X - 1);
    int lane_id = tid & (WARP_SIZE - 1);
    int lane_id_y = (lane_id & 1) + ((lane_id >> 4) << 1); // z-order
    int lane_id_x = (lane_id & 15) >> 1;                   // z-order
    int rewarp_id_y = warp_id_y * WARPDIM_Y + lane_id_y;
    int rewarp_id_x = warp_id_x * WARPDIM_X + lane_id_x;

    __shared__ float smem_a[TILE_K][TILE_M];
    __shared__ float smem_b[TILE_K][TILE_N];

    float reg_a_xx_kk[WORKLOAD_Y];
    float reg_b_kk_xx[WORKLOAD_X];

    for (int ks = 0; ks < k; ks += TILE_K)
    {
        // 使用 BLOCKDIM_Y_A*BLOCKDIM_X_A 个线程加载 TILE_M*TILE_K 个数到 smem_a
        int ida_x = ks + tid_x_a;
#pragma unroll
        for (int rs = 0; rs < TILE_M; rs += BLOCKDIM_Y_A)
        {
            int ida_y = tile_y0 + rs + tid_y_a;
            smem_a[tid_x_a][(rs + tid_y_a) ^ (tid_x_a << 2)] =
                (ida_y < m && ida_x < k) ? A[ida_y * k + ida_x] : 0.0f;
        }

        // 使用 BLOCKDIM_Y_B*BLOCKDIM_X_B 个线程加载 TILE_K*TILE_N 个数到 smem_b
        int idb_y = ks + tid_y_b;
#pragma unroll
        for (int cs = 0; cs < TILE_N; cs += BLOCKDIM_X_B)
        {
            int idb_x = tile_x0 + cs + tid_x_b;
            smem_b[tid_y_b][cs + tid_x_b] =
                (idb_y < k && idb_x < n) ? B[idb_y * n + idb_x] : 0.0f;
        }

        __syncthreads();

        // 对 smem_a 和 smem_b 做矩阵乘
#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk)
        {
#pragma unroll
            for (int idw_y = 0; idw_y < (WORKLOAD_Y >> 2); ++idw_y)
            {
                int ida_x = ((BLOCKDIM_Y_C * idw_y + rewarp_id_y) << 2);
                FLOAT4(reg_a_xx_kk[idw_y << 2]) = FLOAT4(smem_a[kk][ida_x ^ (kk << 2)]);
            }
#pragma unroll
            for (int idw_x = 0; idw_x < (WORKLOAD_X >> 2); ++idw_x)
            {
                int idb_x = (BLOCKDIM_X_C * idw_x + rewarp_id_x) << 2;
                FLOAT4(reg_b_kk_xx[idw_x << 2]) = FLOAT4(smem_b[kk][idb_x]);
            }
#pragma unroll
            for (int idw_y = 0; idw_y < WORKLOAD_Y; ++idw_y)
            {
                for (int idw_x = 0; idw_x < WORKLOAD_X; ++idw_x)
                {
                    workload[idw_y][idw_x] += reg_a_xx_kk[idw_y] * reg_b_kk_xx[idw_x];
                }
            }
        }

        __syncthreads();
    }

    // 最后写入
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
                C[idc_y * n + idc_x] = workload[idw_y][idw_x];
            }
        }
    }
    return;
}

int gemm_v6(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream)
{
    // gemm_v6 聚焦于修改写法
    // 修改写法的过程中, 几乎把整体代码和参考中的代码完全进行了对齐, 但始终无法达到相同的性能,
    // 最后发现, 如果给 float 类型变量赋值使用 0.0, 性能就会很差, 如果用 0.0f 性能就上来了!!! 坑!!!
    // 参考: https://zhuanlan.zhihu.com/p/1910636263666610461
    constexpr int warp_size = 32;
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

    // constexpr int smemSize = tile_m * tile_k + tile_ny* tile_k;
    constexpr int blockDim_y_a = numThreadsPerBlock / blockDim_x_a;
    constexpr int blockDim_y_b = numThreadsPerBlock / blockDim_x_b;
    constexpr int blockDim_y_c = numThreadsPerBlock / blockDim_x_c;
    constexpr int warpDim_y = warp_size / warpDim_x;

    constexpr int workload_x = tile_n / blockDim_x_c;
    constexpr int workload_y = tile_m / blockDim_y_c;

    constexpr int num_warp_x = blockDim_x_c / warpDim_x;
    constexpr int num_warp_y = blockDim_y_c / warpDim_y;

    // gemm_kernel_v6<tile_m, tile_n, tile_k,
    //                blockDim_x_a, blockDim_y_a,
    //                blockDim_x_b, blockDim_y_b,
    //                blockDim_x_c, blockDim_y_c,
    //                workload_x, workload_y,
    //                warpDim_x, warpDim_y,
    //                num_warp_x, num_warp_y,
    //                numThreadsPerBlock, warp_size>
    //     <<<gridDims, blockDims, 0, stream>>>(A, B, C, m, n, k);

    gemm_kernel_v6<<<gridDims, blockDims, 0, stream>>>(A, B, C, m, n, k);
    return 0;
}
