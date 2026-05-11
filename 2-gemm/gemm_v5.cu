#include "gemm_vx.h"

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

template <int TILE_M, int TILE_N, int TILE_K, int BLOCKDIM_X_A, int BLOCKDIM_X_B, int BLOCKDIM_X_C, int WARPDIM_X, int BLOCKDIM>
__global__ void gemm_kernel_v5(const float *A, const float *B, float *C,
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
    int tid_y_c = tid / BLOCKDIM_X_C;
    int tid_x_c = tid % BLOCKDIM_X_C;

    // warp 重组, 一个 warp 内的线程以 [WARPDIM_Y, WARPDIM_X] 的方式组织
    // 一个 block 在 y 方向上有 NUM_WARPS_Y 个 warp, 在 x 方向上有 NUM_WARPS_X 个 warp
    // rewarp_id_y 和 rewarp_id_x 分别是重组后线程在 block 内的 y 和 x 方向的 id
    // 可以使用 rewarp_id_y 和 rewarp_id_x 直接代替 warp 重组之前的 tid_y_c 和 tid_x_c
    constexpr int WARP_SIZE = 32;
    constexpr int WARPDIM_Y = WARP_SIZE / WARPDIM_X;      // 4
    constexpr int NUM_WARPS = BLOCKDIM / WARP_SIZE;       // 8
    constexpr int NUM_WARPS_X = BLOCKDIM_X_C / WARPDIM_X; // 2
    constexpr int NUM_WARPS_Y = BLOCKDIM_Y_C / WARPDIM_Y; // 4
    int warp_id = tid / WARP_SIZE;
    int warp_id_y = warp_id / NUM_WARPS_X;
    int warp_id_x = warp_id % NUM_WARPS_X;
    int lane_id = tid % WARP_SIZE;
    int lane_id_y = lane_id / WARPDIM_X;
    int lane_id_x = lane_id % WARPDIM_X;
    int rewarp_id_y = warp_id_y * WARPDIM_Y + lane_id_y;
    int rewarp_id_x = warp_id_x * WARPDIM_X + lane_id_x;

    extern __shared__ char smem_v3[];
    float *smem_a = (float *)smem_v3;         // [TILE_M, TILE_K]
    float *smem_b = &smem_a[TILE_M * TILE_K]; // [TILE_K, TILE_N]

    for (int ks = 0; ks < k; ks += TILE_K)
    {
        // 使用 BLOCKDIM_Y_A*BLOCKDIM_X_A 个线程加载 TILE_M*TILE_K 个数到 smem_a
        for (int rs = 0; rs < TILE_M; rs += BLOCKDIM_Y_A)
        {
            int idsa_y = rs + tid_y_a;
            int ida_y = tile_y0 + idsa_y;
            for (int cs = 0; cs < TILE_K; cs += BLOCKDIM_X_A)
            {
                int idsa_x = cs + tid_x_a;
                int ida_x = ks + idsa_x;
                smem_a[idsa_y * TILE_K + idsa_x] = (ida_y < m && ida_x < k) ? A[ida_y * k + ida_x] : 0.0;
            }
        }

        // 使用 BLOCKDIM_Y_B*BLOCKDIM_X_B 个线程加载 TILE_K*TILE_N 个数到 smem_b
        for (int rs = 0; rs < TILE_K; rs += BLOCKDIM_Y_B)
        {
            int idsb_y = rs + tid_y_b;
            int idb_y = ks + idsb_y;
            for (int cs = 0; cs < TILE_N; cs += BLOCKDIM_X_B)
            {
                int idsb_x = cs + tid_x_b;
                int idb_x = tile_x0 + idsb_x;
                smem_b[idsb_y * TILE_N + idsb_x] = (idb_y < k && idb_x < n) ? B[idb_y * n + idb_x] : 0.0;
            }
        }

        __syncthreads();

        // 对 smem_a 和 smem_b 做矩阵乘
        // 使用 BLOCKDIM_Y_C*BLOCKDIM_X_C 个线程计算 TILE_M*TILE_N 个数写入 workload, 每个线程计算 WORKLOAD_Y*WORKLOAD_X 个数
        // 外积, 但使用 register
        for (int kk = 0; kk < TILE_K; ++kk)
        {
            float smem_a_xx_kk[WORKLOAD_Y];
            // 原始版本中, 每个线程在循环内加载 smem_a 是按列加载的, 无法使用 float4 进行向量化
            for (int rs = 0, idw_y = 0; rs < TILE_M && idw_y < WORKLOAD_Y; rs += BLOCKDIM_Y_C, ++idw_y)
            {
                smem_a_xx_kk[idw_y] = smem_a[(rs + rewarp_id_y) * TILE_K + kk];
            }

            float smem_b_kk_xx[WORKLOAD_X];
            // 原始版本中, 每个线程在循环内加载 smem_b 是按行加载的, 可以使用 float4 进行向量化
            // for (int cs = 0, idw_x = 0; cs < TILE_N && idw_x < WORKLOAD_X; cs += BLOCKDIM_X_C, ++idw_x)
            // {
            //     smem_b_kk_xx[idw_x] = smem_b[kk * TILE_N + cs + rewarp_id_x];
            // }
            for (int cs = 0, idw_x = 0; cs < TILE_N && idw_x < WORKLOAD_X * 4; cs += BLOCKDIM_X_C * 4, idw_x += 4)
            {
                FLOAT4(smem_b_kk_xx[idw_x]) = FLOAT4(smem_b[kk * TILE_N + cs + rewarp_id_x * 4]);
            }

            for (int idw_y = 0; idw_y < WORKLOAD_Y; ++idw_y)
            {
                for (int idw_x = 0; idw_x < WORKLOAD_X; ++idw_x)
                {
                    workload[idw_y * WORKLOAD_X + idw_x] += smem_a_xx_kk[idw_y] * smem_b_kk_xx[idw_x];
                }
            }
        }

        __syncthreads();
    }

    // 使用 BLOCKDIM_Y_C*BLOCKDIM_X_C 个线程将 BLOCKDIM_Y_C*BLOCKDIM_X_C 个数写入 C
    // 每个线程写入 WORKLOAD_Y * WORKLOAD_X 个数
    for (int rs = 0, idw_y = 0; rs < TILE_M && idw_y < WORKLOAD_Y; rs += BLOCKDIM_Y_C, ++idw_y)
    {
        int idc_y = tile_y0 + rs + rewarp_id_y;
        // for (int cs = 0, idw_x = 0; cs < TILE_N && idw_x < WORKLOAD_X; cs += BLOCKDIM_X_C, ++idw_x)
        for (int cs = 0, idw_x = 0; cs < TILE_N && idw_x < WORKLOAD_X * 4; cs += BLOCKDIM_X_C * 4, idw_x += 4)
        {
            int idc_x = tile_x0 + cs + rewarp_id_x * 4;
            if (idc_y < m && idc_x < n)
            {
                FLOAT4(C[idc_y * n + idc_x]) = FLOAT4(workload[idw_y * WORKLOAD_X + idw_x]);
            }
        }
    }

    return;
}

int gemm_v5(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream)
{
    // gemm_v5 使用 float4 进行向量化访存, 但如果在加载 A 和 B 的时候就使用 float4, 那就要求 A 和 B 的内存地址都是 4 字节对齐的, 这样就失去了通用性
    // 因此, 只在加载 smem_a 和 smem_b 的时候使用 float4
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

    constexpr int smemSize = tile_m * tile_k + tile_n * tile_k;
    gemm_kernel_v5<tile_m, tile_n, tile_k, blockDim_x_a, blockDim_x_b, blockDim_x_c, warpDim_x, numThreadsPerBlock>
        <<<gridDims, blockDims, smemSize * sizeof(float), stream>>>(A, B, C, m, n, k);
    return 0;
}
