#include "gemm_vx.h"

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template <int TILE_M = 128, int TILE_N = 128, int TILE_K = 8,
          int BLOCKDIM_X_A = 8, int BLOCKDIM_Y_A = 32,
          int BLOCKDIM_X_B = 32, int BLOCKDIM_Y_B = 8,
          int BLOCKDIM_X_C = 16, int BLOCKDIM_Y_C = 16,
          int WORKLOAD_X = 8, int WORKLOAD_Y = 8,
          int WARPDIM_X = 8, int WARPDIM_Y = 4,
          int NUM_WARP_X = 2, int NUM_WARP_Y = 4,
          int BLOCKDIM = 256, int WARP_SIZE = 32>
__global__
    // __launch_bounds__(256, 0) // 256线程, 期望每个SM至少2个块
    void gemm_kernel_v7(const float *A, const float *B, float *C,
                        const int m, const int n, const int k)
{
    int tile_y0 = blockIdx.y * TILE_M;
    int tile_x0 = blockIdx.x * TILE_N;

    if (tile_y0 >= m || tile_x0 >= n)
    {
        return;
    }

    // 共享内存
    // smem_a 采用列优先保存(转置)
    __shared__ float smem_a[2][TILE_K][TILE_M];
    __shared__ float smem_b[2][TILE_K][TILE_N];

    // 寄存器
    float reg_a_xx_kk[2][WORKLOAD_Y];
    float reg_b_kk_xx[2][WORKLOAD_X];

    // 每个线程负责计算 C 中的 WORKLOAD_Y*WORKLOAD_X 个数
    float workload[WORKLOAD_Y][WORKLOAD_X] = {};

    // 计算线程在不同组织情况下, x 和 y 向的偏移
    // a % b 求余运算等价于位运算 a & (b - 1)
    int tid = threadIdx.x;
    int tid_y_a = tid / BLOCKDIM_X_A;
    int tid_x_a = tid & (BLOCKDIM_X_A - 1);
    int tid_y_b = tid / BLOCKDIM_X_B;
    int tid_x_b = tid & (BLOCKDIM_X_B - 1);

    // warp 重组
    int warp_id = tid / WARP_SIZE;
    int warp_id_y = warp_id / (NUM_WARP_X);
    int warp_id_x = warp_id & (NUM_WARP_X - 1);
    int lane_id = tid & (WARP_SIZE - 1);
    int lane_id_y = (lane_id & 1) + ((lane_id >> 4) << 1); // z-order
    int lane_id_x = (lane_id & 15) >> 1;                   // z-order
    int rewarp_id_y = warp_id_y * WARPDIM_Y + lane_id_y;
    int rewarp_id_x = warp_id_x * WARPDIM_X + lane_id_x;

    // pre fetch
#pragma unroll
    for (int rs = 0; rs < TILE_M; rs += BLOCKDIM_Y_A)
    {
        int ida_y = tile_y0 + rs + tid_y_a;
        smem_a[0][tid_x_a][(rs + tid_y_a) ^ (tid_x_a << 2)] =
            (ida_y < m && tid_x_a < k) ? A[ida_y * k + tid_x_a] : 0.0f;
    }
#pragma unroll
    for (int cs = 0; cs < TILE_N; cs += BLOCKDIM_X_B)
    {
        int idb_x = tile_x0 + cs + tid_x_b;
        smem_b[0][tid_y_b][cs + tid_x_b] =
            (tid_y_b < k && idb_x < n) ? B[tid_y_b * n + idb_x] : 0.0f;
    }
    __syncthreads();

    // k loop
    int buf_id = 0;
#pragma unroll
    for (int ks = TILE_K; ks < k + TILE_K; ks += TILE_K)
    {
        // workload = smem_a @ smem_b
        // 采用外积 + 寄存器数组 + 向量化访存的方式
#pragma unroll
        for (int kk = 0; kk < TILE_K + 1; ++kk)
        {
            if (kk > 0)
            {
#pragma unroll
                for (int idw_y = 0; idw_y < WORKLOAD_Y; ++idw_y)
                {
                    for (int idw_x = 0; idw_x < WORKLOAD_X; ++idw_x)
                    {
                        workload[idw_y][idw_x] += reg_a_xx_kk[(kk - 1) & 1][idw_y] *
                                                  reg_b_kk_xx[(kk - 1) & 1][idw_x];
                    }
                }
            }

            if (kk < TILE_K)
            {
#pragma unroll
                for (int idw_y = 0; idw_y < (WORKLOAD_Y >> 2); ++idw_y)
                {
                    int ida_x = ((BLOCKDIM_Y_C * idw_y + rewarp_id_y) << 2);
                    FLOAT4(reg_a_xx_kk[kk & 1][idw_y << 2]) =
                        FLOAT4(smem_a[buf_id][kk][ida_x ^ (kk << 2)]);
                }
#pragma unroll
                for (int idw_x = 0; idw_x < (WORKLOAD_X >> 2); ++idw_x)
                {
                    int idb_x = (BLOCKDIM_X_C * idw_x + rewarp_id_x) << 2;
                    FLOAT4(reg_b_kk_xx[kk & 1][idw_x << 2]) =
                        FLOAT4(smem_b[buf_id][kk][idb_x]);
                }
            }
        }

        if (ks < k)
        {
            // 使用1 个 block 加载 TILE_M*TILE_K 个数到 smem_a
            // 只有 y 方向需要迭代, x 方向 BLOCKDIM_X_A == TILE_K
            // 这里用到的 swizzle 方式是: a[y][x] => a[y][x ^ (4 * y)]
            // 注意这里给 float 类型的变量赋值一定要用 0.0f !!!
            int ida_x = ks + tid_x_a;
#pragma unroll
            for (int rs = 0; rs < TILE_M; rs += BLOCKDIM_Y_A)
            {
                int ida_y = tile_y0 + rs + tid_y_a;
                smem_a[buf_id ^ 1][tid_x_a][(rs + tid_y_a) ^ (tid_x_a << 2)] =
                    (ida_y < m && ida_x < k) ? A[ida_y * k + ida_x] : 0.0f;
            }

            // 使用1 个 block 加载 TILE_K*TILE_N 个数到 smem_b
            // 只有 x 方向需要迭代, y 方向 BLOCKDIM_Y_B == TILE_K
            int idb_y = ks + tid_y_b;
#pragma unroll
            for (int cs = 0; cs < TILE_N; cs += BLOCKDIM_X_B)
            {
                int idb_x = tile_x0 + cs + tid_x_b;
                smem_b[buf_id ^ 1][tid_y_b][cs + tid_x_b] =
                    (idb_y < k && idb_x < n) ? B[idb_y * n + idb_x] : 0.0f;
            }

            __syncthreads();
        }

        buf_id ^= 1;
    }

    // 计算结果写回 C 矩阵
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

// double buffering
template <int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
          int A_BLOCK_Y = 32, int B_BLOCK_X = 32, int C_BLOCK_X = 16, int C_BLOCK_Y = 16,
          int C_WARP_X = 8, int C_WARP_Y = 4, int C_WARP_DIM_X = 2, int Tm = 8, int Tn = 8>
__global__ void doublebufferingGEMM(const float *A, const float *B, float *C, const int M, const int K,
                                    const int N)
{
    __shared__ float As[2][Bk][Bm]; // 存储转置后的 tileA
    __shared__ float Bs[2][Bk][Bn]; // 存储 tileB

    // 计算 block 负责的 tileC 左上角元素的行列坐标
    int r0 = blockIdx.y * Bm;
    int c0 = blockIdx.x * Bn;

    // 当前 thread 的编号（默认为一维 block 配置）
    int tid = threadIdx.x;

    /*------ tileA ------*/
    // 对于 tid 号线程，其位于 blockA 中的行列坐标为 (tid / A_BLOCK_X, tid % A_BLOCK_X)
    int A_THREAD_Y = tid / A_BLOCK_X;
    int A_THREAD_X = tid & (A_BLOCK_X - 1);

    /*------ tileB ------*/
    // 对于 tid 号线程，其位于 blockB 中的行列坐标为 (tid / B_BLOCK_X, tid % B_BLOCK_X)
    int B_THREAD_Y = tid / B_BLOCK_X;
    int B_THREAD_X = tid & (B_BLOCK_X - 1);

    // 按 8*4 排列 warp

    // 计算当前 thread 属于哪个 warp，第几个 lane
    constexpr int WARP_SIZE = 32;
    int warpId = tid / WARP_SIZE;
    int laneId = tid & (WARP_SIZE - 1);

    // 计算当前 thread 所在 warp 在 block 中的 x, y 坐标
    int warpX = warpId & (C_WARP_DIM_X - 1);
    int warpY = warpId / C_WARP_DIM_X;

    // z-order 排布，计算当前 thread 在 warp 中的 x, y 坐标
    int laneY = (laneId & 1) + ((laneId >> 4) << 1);
    int laneX = (laneId & 15) >> 1;

    // 当前 thread 在 blockC 中的行列坐标为 (warpY * C_WARP_Y + laneY, warpX * C_WARP_X + laneX)
    int C_THREAD_Y = warpY * C_WARP_Y + laneY;
    int C_THREAD_X = warpX * C_WARP_X + laneX;

    // 每个 thread 负责 Tm * Tn 个元素计算
    float Ct[Tm][Tn] = {0.0};

    // 存储 A 中列向量和 B 中行向量
    float regA[2][Tm] = {0.0f};
    float regB[2][Tn] = {0.0f};

    int buffer_id = 0;

    // 预先读取 k = 0 数据
#pragma unroll
    for (int i = 0; i < Bm; i += A_BLOCK_Y)
    {
        int r = r0 + i + A_THREAD_Y;
        As[0][A_THREAD_X][(i + A_THREAD_Y) ^ (A_THREAD_X << 2)] =
            (r < M && A_THREAD_X < K) ? A[r * K + A_THREAD_X] : 0.0f;
    }

#pragma unroll
    for (int j = 0; j < Bm; j += B_BLOCK_X)
    {
        int c = c0 + j + B_THREAD_X;
        Bs[0][B_THREAD_Y][j + B_THREAD_X] = (B_THREAD_Y < K && c < N) ? B[B_THREAD_Y * N + c] : 0.0f;
    }

    __syncthreads();

    // K-Loop，跳过 k = 0 并且最后增加一次循环
    for (int k = Bk; k < K + Bk; k += Bk)
    {
        // 计算阶段：计算第 k - 1 次循环结果
        // 双缓冲 p-Loop
#pragma unroll
        for (int p = 0; p < Bk + 1; ++p)
        {
            // 计算阶段：计算第 p - 1 次循环的结果
            if (p > 0)
            {
#pragma unroll
                for (int i = 0; i < Tm; ++i)
                {
#pragma unroll
                    for (int j = 0; j < Tn; ++j)
                    {
                        Ct[i][j] += regA[(p - 1) & 1][i] * regB[(p - 1) & 1][j];
                    }
                }
            }

            // 预取阶段：读取第 p 次循环的数据
            if (p < Bk)
            {
                // 读取 regA
#pragma unroll
                for (int i = 0; i < (Tm >> 2); ++i)
                {
                    int r = (C_THREAD_Y + i * C_BLOCK_Y) << 2;
                    FLOAT4(regA[p & 1][i << 2]) = FLOAT4(As[buffer_id][p][r ^ (p << 2)]);
                }

                // 读取 regB
#pragma unroll
                for (int j = 0; j < (Tn >> 2); ++j)
                {
                    int c = (C_THREAD_X + j * C_BLOCK_X) << 2;
                    FLOAT4(regB[p & 1][j << 2]) = FLOAT4(Bs[buffer_id][p][c]);
                }
            }
        }

        // 预取阶段：预取第 k 次循环数据
        if (k < K)
        {
            // 读取 tileA
            int c = k + A_THREAD_X;
#pragma unroll
            for (int i = 0; i < Bm; i += A_BLOCK_Y)
            {
                int r = r0 + i + A_THREAD_Y;
                As[buffer_id ^ 1][A_THREAD_X][(i + A_THREAD_Y) ^ (A_THREAD_X << 2)] =
                    (r < M && c < K) ? A[r * K + c] : 0.f; // 转置
            }

            // 读取 tileA
            int r = k + B_THREAD_Y;
#pragma unroll
            for (int j = 0; j < Bn; j += B_BLOCK_X)
            {
                c = c0 + j + B_THREAD_X;
                Bs[buffer_id ^ 1][B_THREAD_Y][j + B_THREAD_X] = (r < K && c < N) ? B[r * N + c] : 0.f;
            }

            __syncthreads();
        }

        buffer_id ^= 1; // 切换缓冲区
    }

    // 将 Ct 写入 C
#pragma unroll
    for (int i = 0; i < Tm; ++i)
    {
        int r = r0 + (C_THREAD_Y << 2) + ((i >> 2) << 2) * C_BLOCK_Y + (i & 3);
#pragma unroll
        for (int j = 0; j < Tn; ++j)
        {
            int c = c0 + (C_THREAD_X << 2) + ((j >> 2) << 2) * C_BLOCK_X + (j & 3);

            if (r < M && c < N)
            {
                C[r * N + c] = Ct[i][j];
            }
        }
    }
}

int gemm_v7(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream)
{
    // gemm_v7 double buffer
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

    gemm_kernel_v7<<<gridDims, blockDims, 0, stream>>>(A, B, C, m, n, k);
    return 0;
}

int gemm_v7_(const float *A, const float *B, float *C,
             int m, int n, int k, cudaStream_t stream)
{
    // gemm_v7 double buffer
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

    doublebufferingGEMM<<<gridDims, blockDims, 0, stream>>>(A, B, C, m, k, n);
    return 0;
}
