#include "bmm_vx.h"

template <int WARPSIZE, int BLOCKSIZE,
          int TILEM, int TILEN, int TILEK,
          int BLOCKROWA, int BLOCKCOLA,
          int BLOCKROWB, int BLOCKCOLB,
          int BLOCKROWC, int BLOCKCOLC>
__global__ void bmm_kernel_v1(const float *A, const float *B, float *C,
                              int bs, int m, int n, int k)
{
    int batchStartA = blockIdx.z * m * k;
    int batchStartB = blockIdx.z * n * k;
    int batchStartC = blockIdx.z * m * n;

    int row0 = blockIdx.y * TILEM;
    int col0 = blockIdx.x * TILEN;

    __shared__ float sharedTileA[TILEM][TILEK];
    __shared__ float sharedTileB[TILEK][TILEN];
    float regSum[TILEM / BLOCKROWC][TILEN / BLOCKCOLC] = {0.0f};

    // Method 2
    // float regA[TILEM / BLOCKROWC][TILEK] = {0.0f};
    // float regB[TILEN / BLOCKCOLC][TILEK] = {0.0f};
    // Method 3
    float regA[TILEM / BLOCKROWC] = {0.0f};
    float regB[TILEN / BLOCKCOLC] = {0.0f};

    int tId = threadIdx.x;
    int rowIdA = tId / BLOCKCOLA;
    int colIdA = tId % BLOCKCOLA;
    int rowIdB = tId / BLOCKCOLB;
    int colIdB = tId % BLOCKCOLB;
    int rowIdC = tId / BLOCKCOLC;
    int colIdC = tId % BLOCKCOLC;

    // k loop
    for (int kk = 0; kk < k; kk += TILEK)
    {
        for (int ii = 0; ii < TILEM; ii += BLOCKROWA)
        {
            for (int jj = 0; jj < TILEK; jj += BLOCKCOLA)
            {
                sharedTileA[ii + rowIdA][jj + colIdA] =
                    row0 + ii + rowIdA < m && kk + jj + colIdA < k ?
                        A[batchStartA + (row0 + ii + rowIdA) * k + (kk + jj + colIdA)] : 0.0f;
            }
        }
        for (int ii = 0; ii < TILEK; ii += BLOCKROWB)
        {
            for (int jj = 0; jj < TILEN; jj += BLOCKCOLB)
            {
                sharedTileB[ii + rowIdB][jj + colIdB] =
                    kk + ii + rowIdB < k && col0 + jj + colIdB < n ?
                        B[batchStartB + (kk + ii + rowIdB) * n + (col0 + jj + colIdB)] : 0.0f;
            }
        }
        __syncthreads();

        // Method 1
        // for (int tk = 0; tk < TILEK; ++tk)
        // {
        //     for (int ii = 0; ii < TILEM; ii += BLOCKROWC)
        //     {
        //         for (int jj = 0; jj < TILEN; jj += BLOCKCOLC)
        //         {
        //             regSum[ii / BLOCKROWC][jj / BLOCKCOLC] +=
        //                 sharedTileA[ii + rowIdC][tk] * sharedTileB[tk][jj + colIdC];
        //         }
        //     }
        // }

        // Method 2
        // for (int ii = 0; ii < TILEM; ii += BLOCKROWC)
        // {
        //     for (int tk = 0; tk < TILEK; ++tk)
        //     {
        //         regA[ii / BLOCKROWC][tk] = sharedTileA[ii + rowIdC][tk];
        //     }
        // }

        // for (int ii = 0; ii < TILEN; ii += BLOCKCOLC)
        // {
        //     for (int tk = 0; tk < TILEK; ++tk)
        //     {
        //         regB[ii / BLOCKCOLC][tk] = sharedTileB[tk][ii + colIdC];
        //     }
        // }

        // for (int tk = 0; tk < TILEK; ++tk)
        // {
        //     for (int ii = 0; ii < TILEM; ii += BLOCKROWC)
        //     {
        //         for (int jj = 0; jj < TILEN; jj += BLOCKCOLC)
        //         {
        //             regSum[ii / BLOCKROWC][jj / BLOCKCOLC] += regA[ii / BLOCKROWC][tk] * regB[ii / BLOCKCOLC][tk];
        //         }
        //     }
        // }

        // Method 3
        for (int tk = 0; tk < TILEK; ++tk)
        {
            for (int ii = 0; ii < TILEM; ii += BLOCKROWC) {
                regA[ii / BLOCKROWC] = sharedTileA[ii + rowIdC][tk];
            }

            for (int ii = 0; ii < TILEN; ii += BLOCKCOLC) {
                regB[ii / BLOCKCOLC] = sharedTileB[tk][ii + colIdC];
            }
            
            for (int ii = 0; ii < TILEM; ii += BLOCKROWC)
            {
                for (int jj = 0; jj < TILEN; jj += BLOCKCOLC)
                {
                    regSum[ii / BLOCKROWC][jj / BLOCKCOLC] += regA[ii / BLOCKROWC] * regB[jj / BLOCKCOLC];
                }
            }
        }
        __syncthreads();
    }

    for (int ii = 0; ii < TILEM; ii += BLOCKROWC)
    {
        for (int jj = 0; jj < TILEN; jj += BLOCKCOLC)
        {
            if (row0 + ii + rowIdC < m && col0 + jj + colIdC < n)
            {
                C[batchStartC + (row0 + ii + rowIdC) * n + (col0 + jj + colIdC)] =
                    regSum[ii / BLOCKROWC][jj / BLOCKCOLC];
            }
        }
    }

    return;
}

int bmm_v2(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32;
    const int blockSize = 256;
    dim3 blockDims(blockSize);

    const int tileM = 128;
    const int tileN = 128;
    const int tileK = 8;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK;
    const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN;
    const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16;
    const int blockColC = 16;

    /**
     * 单纯分析全局内存的计算访存比:
     *   全局内存访存次数: tileM * k + tileN * k
     *   计算次数: tileM * tileN + tileM * tileN
     * 最大化计算访存比需要 tileM 和 tileN 相互接近并尽可能大;
     * 但同时不能太大导致共享内存和寄存器超限;
     * v1 中的实验显示, tileM = tileN = 128, tileK = 8 的性能较好;
     * ---------------------------------------------------------------------------------
     * 观察 method 1, method 2 和 method 3, 共享内存的计算访存比为:
     *   共享内存访问次数: (TILEM / BLOCKROWC) + (TILEN / BLOCKCOLC)
     *   计算次数: (TILEM / BLOCKROWC) * (TILEN / BLOCKCOLC)
     * 想让访存比最大化, 则需要 TILEM / BLOCKROWC 和 TILEN / BLOCKCOLC 相互接近并且尽可能大;
     * 所以 blockRowC = blockColC = sqrt(blockSize)
     * ---------------------------------------------------------------------------------
     * 相较于 method 2, method 3 在相同计算访存比的条件下, 使用了更少的寄存器;
     */

    int gridDim_z = bs;
    int gridDim_y = (m + tileM - 1) / tileM;
    int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}
