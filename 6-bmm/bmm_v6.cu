#include "bmm_vx.h"

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CONST_FLOAT4(value) (reinterpret_cast<const float4 *>(&(value))[0])

template <int WARPSIZE, int BLOCKSIZE,
          int TILEM, int TILEN, int TILEK,
          int blockLayoutRowA, int blockLayoutColA,
          int blockLayoutRowB, int blockLayoutColB,
          int blockLayoutRowC, int blockLayoutColC,
          int warpLayoutRowA, int warpLayoutColA,
          int warpLayoutRowB, int warpLayoutColB,
          int warpLayoutRowC, int warpLayoutColC>
__global__ void bmm_kernel_v6(const float *A, const float *B, float *C,
                              int bs, int m, int n, int k)
{
    int batchStartA = blockIdx.z * m * k;
    int batchStartB = blockIdx.z * n * k;
    int batchStartC = blockIdx.z * m * n;

    int row0 = blockIdx.y * TILEM;
    int col0 = blockIdx.x * TILEN;

    __shared__ float sharedTileA[2][TILEK][TILEM];
    __shared__ float sharedTileB[2][TILEK][TILEN];

    float regSum[TILEM / blockLayoutRowC][TILEN / blockLayoutColC] = {0.0f};
    float regA[TILEM / blockLayoutRowC] = {0.0f};
    float regB[TILEN / blockLayoutColC] = {0.0f};

    int tId = threadIdx.x;
    int warpId = tId / WARPSIZE;
    int laneId = tId % WARPSIZE;

    [[maybe_unused]] const int numWarpsRowA = blockLayoutRowA / warpLayoutRowA;
    const int numWarpsColA = blockLayoutColA / warpLayoutColA;
    int warpIdRowA = warpId / numWarpsColA;
    int warpIdColA = warpId % numWarpsColA;
    int laneIdRowA = laneId / warpLayoutColA;
    int laneIdColA = laneId % warpLayoutColA;
    int rowIdA = warpIdRowA * warpLayoutRowA + laneIdRowA;
    int colIdA = warpIdColA * warpLayoutColA + laneIdColA;

    [[maybe_unused]] const int numWarpsRowB = blockLayoutRowB / warpLayoutRowB;
    const int numWarpsColB = blockLayoutColB / warpLayoutColB;
    int warpIdRowB = warpId / numWarpsColB;
    int warpIdColB = warpId % numWarpsColB;
    int laneIdRowB = laneId / warpLayoutColB;
    int laneIdColB = laneId % warpLayoutColB;
    int rowIdB = warpIdRowB * warpLayoutRowB + laneIdRowB;
    int colIdB = warpIdColB * warpLayoutColB + laneIdColB;

    [[maybe_unused]] const int numWarpsRowC = blockLayoutRowC / warpLayoutRowC;
    const int numWarpsColC = blockLayoutColC / warpLayoutColC;
    int warpIdRowC = warpId / numWarpsColC;
    int warpIdColC = warpId % numWarpsColC;
    // int laneIdRowC = laneId / warpLayoutColC;
    // int laneIdColC = laneId % warpLayoutColC;
    int laneIdRowC = laneId / 16 * 2 + laneId % 2; // z-order
    int laneIdColC = laneId % 16 / 2;              // z-order
    int rowIdC = warpIdRowC * warpLayoutRowC + laneIdRowC;
    int colIdC = warpIdColC * warpLayoutColC + laneIdColC;
    int physical_rowIdC = rowIdC << 2;
    int physical_colIdC = colIdC << 2;

    // k loop
    for (int kk = 0; kk < k + TILEK; kk += TILEK)
    {
        int bufId0 = (kk / TILEK) % 2;
        int bufId1 = ((kk + 1) / TILEK) % 2;
        if (kk < k)
        {
            for (int ii = 0; ii < TILEM; ii += blockLayoutRowA)
            {
                int logical_rowIdA = 0; // swizzle
                int physical_rowIdA = ii + rowIdA;
                int physical_colIdA = colIdA << 2;
                // x + y % 8 / 4 * 4 => x + (y & 4)
                if (k % 4 == 0 && row0 + ii + rowIdA < m && kk + physical_colIdA + 3 < k)
                {
                    float4 tmp = CONST_FLOAT4(A[batchStartA + (row0 + physical_rowIdA) * k + (kk + physical_colIdA)]);
                    logical_rowIdA = (physical_rowIdA + ((physical_colIdA + 0) & 4)) & (TILEM - 1);
                    sharedTileA[bufId0][physical_colIdA + 0][logical_rowIdA] = tmp.x;
                    logical_rowIdA = (physical_rowIdA + ((physical_colIdA + 1) & 4)) & (TILEM - 1);
                    sharedTileA[bufId0][physical_colIdA + 1][logical_rowIdA] = tmp.y;
                    logical_rowIdA = (physical_rowIdA + ((physical_colIdA + 2) & 4)) & (TILEM - 1);
                    sharedTileA[bufId0][physical_colIdA + 2][logical_rowIdA] = tmp.z;
                    logical_rowIdA = (physical_rowIdA + ((physical_colIdA + 3) & 4)) & (TILEM - 1);
                    sharedTileA[bufId0][physical_colIdA + 3][logical_rowIdA] = tmp.w;
                }
                else
                {
                    for (int fi = 0; fi < 4; ++fi)
                    {
                        logical_rowIdA = (physical_rowIdA + ((physical_colIdA + fi) & 4)) & (TILEM - 1);
                        sharedTileA[bufId0][physical_colIdA + fi][logical_rowIdA] =
                            row0 + physical_rowIdA < m && kk + physical_colIdA + fi < k ? A[batchStartA + (row0 + physical_rowIdA) * k + (kk + physical_colIdA + fi)] : 0.0f;
                    }
                }
            }
            for (int ii = 0; ii < TILEK; ii += blockLayoutRowB)
            {
                int physical_rowIdB = ii + rowIdB;
                int physical_colIdB = colIdB << 2;
                if (n % 4 == 0 && kk + physical_rowIdB < k && col0 + physical_colIdB + 3 < n)
                {
                    FLOAT4(sharedTileB[bufId0][physical_rowIdB][physical_colIdB]) =
                        CONST_FLOAT4(B[batchStartB + (kk + physical_rowIdB) * n + (col0 + physical_colIdB)]);
                }
                else
                {
                    for (int fi = 0; fi < 4; ++fi)
                    {
                        // 如果使用 physical_colIdB + fi, 1*32 的 warp 中, 每个线程的写入地址相差 4 个 word, 会出现 bank conflict;
                        // 通过将 physical_colIdB + fi 改为 colIdB + fi * 32, 使得 1*32 的 warp 写入地址连续的 sharedTileB;
                        int offset = fi << 5;
                        sharedTileB[bufId0][physical_rowIdB][colIdB + offset] =
                            kk + physical_rowIdB < k && col0 + colIdB + offset < n ? B[batchStartB + (kk + physical_rowIdB) * n + (col0 + colIdB + offset)] : 0.0f;
                    }
                }
            }
        }

        if (kk > 0)
        {
            for (int tk = 0; tk < TILEK; ++tk)
            {
                for (int ii = 0; ii < TILEM; ii += blockLayoutRowC << 2)
                {
                    int logical_rowIdC = (ii + physical_rowIdC + (tk & 4)) & (TILEM - 1); // swizzle
                    FLOAT4(regA[ii / blockLayoutRowC]) = FLOAT4(sharedTileA[bufId1][tk][logical_rowIdC]);
                }

                for (int ii = 0; ii < TILEN; ii += blockLayoutColC << 2)
                {
                    FLOAT4(regB[ii / blockLayoutColC]) = FLOAT4(sharedTileB[bufId1][tk][ii + (colIdC << 2)]);
                }

                for (int ii = 0; ii < TILEM; ii += blockLayoutRowC)
                {
                    for (int jj = 0; jj < TILEN; jj += blockLayoutColC)
                    {
                        regSum[ii / blockLayoutRowC][jj / blockLayoutColC] += regA[ii / blockLayoutRowC] * regB[jj / blockLayoutColC];
                    }
                }
            }
        }
        __syncthreads();
    }

    // store
    for (int ii = 0; ii < TILEM; ii += blockLayoutRowC << 2)
    {
        for (int jj = 0; jj < TILEN; jj += blockLayoutColC << 2)
        {
            for (int fi = 0; fi < 4; ++fi)
            {
                if (n % 4 == 0 && row0 + ii + physical_rowIdC + fi < m && col0 + jj + physical_colIdC + 3 < n)
                {
                    FLOAT4(C[batchStartC + (row0 + ii + physical_rowIdC + fi) * n + (col0 + jj + physical_colIdC)]) =
                        FLOAT4(regSum[ii / blockLayoutRowC + fi][jj / blockLayoutColC]);
                }
                else
                {
                    for (int fj = 0; fj < 4; ++fj)
                    {
                        if (row0 + ii + physical_rowIdC + fi < m && col0 + jj + physical_colIdC + fj < n)
                        {
                            C[batchStartC + (row0 + ii + physical_rowIdC + fi) * n + (col0 + jj + physical_colIdC + fj)] =
                                regSum[ii / blockLayoutRowC + fi][jj / blockLayoutColC + fj];
                        }
                    }
                }
            }
        }
    }

    return;
}

int bmm_v6(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32;
    const int blockSize = 256;
    dim3 blockDims(blockSize);

    const int tileM = 128;
    const int tileN = 128;
    const int tileK = 8;
    // block 在处理 sharedTileA 时的布局;
    const int blockLayoutRowA = blockSize / (tileK / 4);   // 128
    const int blockLayoutColA = tileK / 4;                 // 2
    const int warpLayoutRowA = warpSize / blockLayoutColA; // 16
    const int warpLayoutColA = blockLayoutColA;            // 2
    // block 在处理 sharedTileB 时的布局;
    const int blockLayoutRowB = blockSize / (tileN / 4);   // 8
    const int blockLayoutColB = tileN / 4;                 // 32
    const int warpLayoutRowB = warpSize / blockLayoutColB; // 1
    const int warpLayoutColB = blockLayoutColB;            // 32
    // block 在处理 TileC 时的布局;
    const int blockLayoutRowC = 16;
    const int blockLayoutColC = 16;
    // warpLayoutRowC <= blockLayoutRowC
    // warpLayoutColC <= blockLayoutColC
    // warpLayoutRowC * warpLayoutColC = warpSize
    const int warpLayoutRowC = 4;
    const int warpLayoutColC = 8;

    int gridDim_z = bs;
    int gridDim_y = (m + tileM - 1) / tileM;
    int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v6<warpSize, blockSize, tileM, tileN, tileK,
                  blockLayoutRowA, blockLayoutColA, blockLayoutRowB, blockLayoutColB, blockLayoutRowC, blockLayoutColC,
                  warpLayoutRowA, warpLayoutColA, warpLayoutRowB, warpLayoutColB, warpLayoutRowC, warpLayoutColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}
