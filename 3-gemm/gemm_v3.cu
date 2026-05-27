#include "gemm_vx.h"

template <int TILE_M, int TILE_N, int TILE_K, int BLOCKDIM_X_A, int BLOCKDIM_X_B, int BLOCKDIM_X_C, int BLOCKDIM>
__global__ void gemm_kernel_v3(const float *A, const float *B, float *C,
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

    extern __shared__ char smem[];
    float *smem_a = (float *)smem;         // [TILE_M, TILE_K]
    float *smem_b = &smem_a[TILE_M * TILE_K]; // [TILE_K, TILE_N]

    // 计算外积用到的 register
    float smem_a_kk_xx[WORKLOAD_X];
    float smem_a_xx_kk[WORKLOAD_Y];

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

        // 对 smem_a 和 smem_b 做矩阵乘, 这里分别实现内积外积两种方式, 实测内积要快一些, 不明白为什么
        // 使用 BLOCKDIM_Y_C*BLOCKDIM_X_C 个线程计算 TILE_M*TILE_N 个数写入 smem_a 和 smem_b

        // 内积
        // for (int rs = 0, idw_y = 0; rs < TILE_M && idw_y < WORKLOAD_Y; rs += BLOCKDIM_Y_C, ++idw_y)
        // {
        //     for (int cs = 0, idw_x = 0; cs < TILE_N && idw_x < WORKLOAD_X; cs += BLOCKDIM_X_C, ++idw_x)
        //     {
        //         for (int kk = 0; kk < TILE_K; ++kk)
        //         {
        //             workload[idw_y * WORKLOAD_X + idw_x] += smem_a[(rs + tid_y_c) * TILE_K + kk] *
        //                                                     smem_b[kk * TILE_N + cs + tid_x_c];
        //         }
        //     }
        // }

        // 外积
        // for (int kk = 0; kk < TILE_K; ++kk)
        // {
        //     for (int rs = 0, idw_y = 0; rs < TILE_M && idw_y < WORKLOAD_Y; rs += BLOCKDIM_Y_C, ++idw_y)
        //     {
        //         float smem_a_rs_kk = smem_a[(rs + tid_y_c) * TILE_K + kk];
        //         for (int cs = 0, idw_x = 0; cs < TILE_N && idw_x < WORKLOAD_X; cs += BLOCKDIM_X_C, ++idw_x)
        //         {
        //             workload[idw_y * WORKLOAD_X + idw_x] += smem_a_rs_kk * smem_b[kk * TILE_N + cs + tid_x_c];
        //         }
        //     }
        // }

        // 外积, 但使用 register
        for (int kk = 0; kk < TILE_K; ++kk)
        {
            for (int rs = 0, idw_y = 0; rs < TILE_M && idw_y < WORKLOAD_Y; rs += BLOCKDIM_Y_C, ++idw_y)
            {
                smem_a_xx_kk[idw_y] = smem_a[(rs + tid_y_c) * TILE_K + kk];
            }

            for (int cs = 0, idw_x = 0; cs < TILE_N && idw_x < WORKLOAD_X; cs += BLOCKDIM_X_C, ++idw_x)
            {
                smem_a_kk_xx[idw_x] = smem_b[kk * TILE_N + cs + tid_x_c];
            }

            for (int idw_y = 0; idw_y < WORKLOAD_Y; ++idw_y)
            {
                for (int idw_x = 0; idw_x < WORKLOAD_X; ++idw_x)
                {
                    workload[idw_y * WORKLOAD_X + idw_x] += smem_a_xx_kk[idw_y] * smem_a_kk_xx[idw_x];
                }
            }
        }

        __syncthreads();
    }

    // 使用 BLOCKDIM_Y_C*BLOCKDIM_X_C 个线程将 BLOCKDIM_Y_C*BLOCKDIM_X_C 个数写入 C
    // 每个线程写入 WORKLOAD_Y * WORKLOAD_X 个数
    for (int rs = 0, idw_y = 0; rs < TILE_M && idw_y < WORKLOAD_Y; rs += BLOCKDIM_Y_C, ++idw_y)
    {
        int idc_y = tile_y0 + rs + tid_y_c;
        for (int cs = 0, idw_x = 0; cs < TILE_N && idw_x < WORKLOAD_X; cs += BLOCKDIM_X_C, ++idw_x)
        {
            int idc_x = tile_x0 + cs + tid_x_c;
            if (idc_y < m && idc_x < n)
            {
                C[idc_y * n + idc_x] = workload[idw_y * WORKLOAD_X + idw_x];
            }
        }
    }

    return;
}

int gemm_v3(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream)
{
    // gemm_v3 的优点在于, 可以根据 ABC 大小来灵活调整 tile 的大小和 block 的大小
    // 使用 1024 个线程, 计算 C 中的 128 * 64 个点
    // constexpr int tile_m = 128;
    // constexpr int tile_n = 64;
    // constexpr int tile_k = 32;

    // 使用 1 维线程块, 在 kernel 内部根据 tile 大小重新组织 block
    // 这里定义 block 在读写 ABC 时的 x 方向的大小, 由于 block 大小固定, 所以 y 方向的大小也可以确定
    // 这里为了访存合并和避免 bank 冲突, x 方向的大小均设置为 32 的倍数
    // constexpr int blockDim_x_a = 32;
    // constexpr int blockDim_x_b = 64;
    // constexpr int blockDim_x_c = 64;
    // constexpr int numThreadsPerBlock = 1024;

    // !!!为什么这组配置要比上面配置的性能好那么多???
    // 之所以下面的配置性能更好, 是基于计算访存比分析之后得到的结果:
    // 在不考虑合并访存的情况下, tile_m 和 tile_n 越大, 计算访存比越高, 每个线程的计算量越大, 用到的寄存器, shared memory 越多,
    // 这在 v0 以及 v1 的版本中是有利的, 但如果 tile_m 和 tile_n 太大, 回导致 block 数量过度变少, 以及寄存器和 shared memory 的使用过多,
    // 从而导致并行度过低, 无法充分利用 GPU 的计算资源, 反而性能下降;
    // 前人的实验结果表明, tile_m = 128 tile_n = 128 tile_k = 8 是一个比较好的配置;
    // numThreadsPerBlock不能太大, 也是处于相同的考量;
    constexpr int tile_m = 128;
    constexpr int tile_n = 128;
    constexpr int tile_k = 8;
    constexpr int blockDim_x_a = 8;
    constexpr int blockDim_x_b = 32;
    constexpr int blockDim_x_c = 16;
    constexpr int numThreadsPerBlock = 256;
    dim3 blockDims(numThreadsPerBlock);

    int gridDim_x = (n + tile_n - 1) / (tile_n);
    int gridDim_y = (m + tile_m - 1) / (tile_m);
    dim3 gridDims(gridDim_x, gridDim_y);

    constexpr int smemSize = tile_m * tile_k + tile_n * tile_k;
    gemm_kernel_v3<tile_m, tile_n, tile_k, blockDim_x_a, blockDim_x_b, blockDim_x_c, numThreadsPerBlock>
        <<<gridDims, blockDims, smemSize * sizeof(float), stream>>>(A, B, C, m, n, k);
    return 0;
}
