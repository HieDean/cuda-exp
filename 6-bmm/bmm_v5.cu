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
__global__ void bmm_kernel_v5(const float *A, const float *B, float *C,
                              int bs, int m, int n, int k)
{
    int batchStartA = blockIdx.z * m * k;
    int batchStartB = blockIdx.z * n * k;
    int batchStartC = blockIdx.z * m * n;

    int row0 = blockIdx.y * TILEM;
    int col0 = blockIdx.x * TILEN;

    __shared__ float sharedTileA[TILEK][TILEM];
    __shared__ float sharedTileB[TILEK][TILEN];

    float regSum[TILEM / blockLayoutRowC][TILEN / blockLayoutColC] = {0.0f};
    float regA[TILEM / blockLayoutRowC] = {0.0f};
    float regB[TILEN / blockLayoutColC] = {0.0f};

    int tId = threadIdx.x;
    int warpId = tId / WARPSIZE;
    int laneId = tId % WARPSIZE;

    [[maybe_unused]] int numWarpsRowA = blockLayoutRowA / warpLayoutRowA;
    int numWarpsColA = blockLayoutColA / warpLayoutColA;
    int warpIdRowA = warpId / numWarpsColA;
    int warpIdColA = warpId % numWarpsColA;
    int laneIdRowA = laneId / warpLayoutColA;
    int laneIdColA = laneId % warpLayoutColA;
    int rowIdA = warpIdRowA * warpLayoutRowA + laneIdRowA;
    int colIdA = warpIdColA * warpLayoutColA + laneIdColA;

    [[maybe_unused]] int numWarpsRowB = blockLayoutRowB / warpLayoutRowB;
    int numWarpsColB = blockLayoutColB / warpLayoutColB;
    int warpIdRowB = warpId / numWarpsColB;
    int warpIdColB = warpId % numWarpsColB;
    int laneIdRowB = laneId / warpLayoutColB;
    int laneIdColB = laneId % warpLayoutColB;
    int rowIdB = warpIdRowB * warpLayoutRowB + laneIdRowB;
    int colIdB = warpIdColB * warpLayoutColB + laneIdColB;

    [[maybe_unused]] int numWarpsRowC = blockLayoutRowC / warpLayoutRowC;
    int numWarpsColC = blockLayoutColC / warpLayoutColC;
    int warpIdRowC = warpId / numWarpsColC;
    int warpIdColC = warpId % numWarpsColC;
    int laneIdRowC = laneId / warpLayoutColC;
    int laneIdColC = laneId % warpLayoutColC;
    int rowIdC = warpIdRowC * warpLayoutRowC + laneIdRowC;
    int colIdC = warpIdColC * warpLayoutColC + laneIdColC;

    // k loop
    for (int kk = 0; kk < k; kk += TILEK)
    {
        for (int ii = 0; ii < TILEM; ii += blockLayoutRowA)
        {
            for (int jj = 0; jj < TILEK; jj += blockLayoutColA * 4)
            {
                if (k % 4 == 0 && row0 + ii + rowIdA < m && kk + jj + colIdA * 4 + 3 < k)
                {
                    float4 tmp = CONST_FLOAT4(A[batchStartA + (row0 + ii + rowIdA) * k + (kk + jj + colIdA * 4)]);
                    sharedTileA[jj + colIdA * 4 + 0][ii + rowIdA] = tmp.x;
                    sharedTileA[jj + colIdA * 4 + 1][ii + rowIdA] = tmp.y;
                    sharedTileA[jj + colIdA * 4 + 2][ii + rowIdA] = tmp.z;
                    sharedTileA[jj + colIdA * 4 + 3][ii + rowIdA] = tmp.w;
                }
                else
                {
                    for (int fi = 0; fi < 4; ++fi)
                    {
                        sharedTileA[jj + colIdA * 4 + fi][ii + rowIdA] =
                            row0 + ii + rowIdA < m && kk + jj + colIdA * 4 + fi < k ? A[batchStartA + (row0 + ii + rowIdA) * k + (kk + jj + colIdA * 4 + fi)] : 0.0f;
                    }
                }
            }
        }
        for (int ii = 0; ii < TILEK; ii += blockLayoutRowB)
        {
            for (int jj = 0; jj < TILEN; jj += blockLayoutColB * 4)
            {
                if (n % 4 == 0 && kk + ii + rowIdB < k && col0 + jj + colIdB * 4 + 3 < n)
                {
                    FLOAT4(sharedTileB[ii + rowIdB][jj + colIdB * 4]) = CONST_FLOAT4(B[batchStartB + (kk + ii + rowIdB) * n + (col0 + jj + colIdB * 4)]);
                }
                else
                {
                    for (int fi = 0; fi < 4; ++fi)
                    {
                        sharedTileB[ii + rowIdB][jj + colIdB * 4 + fi] =
                            kk + ii + rowIdB < k && col0 + jj + colIdB * 4 + fi < n ? B[batchStartB + (kk + ii + rowIdB) * n + (col0 + jj + colIdB * 4 + fi)] : 0.0f;
                    }
                }
            }
        }
        __syncthreads();

        for (int tk = 0; tk < TILEK; ++tk)
        {
            for (int ii = 0; ii < TILEM; ii += blockLayoutRowC) {
                regA[ii / blockLayoutRowC] = sharedTileA[tk][ii + rowIdC];
            }

            for (int ii = 0; ii < TILEN; ii += blockLayoutColC) {
                regB[ii / blockLayoutColC] = sharedTileB[tk][ii + colIdC];
            }
            
            for (int ii = 0; ii < TILEM; ii += blockLayoutRowC)
            {
                for (int jj = 0; jj < TILEN; jj += blockLayoutColC)
                {
                    regSum[ii / blockLayoutRowC][jj / blockLayoutColC] += regA[ii / blockLayoutRowC] * regB[jj / blockLayoutColC];
                }
            }
        }
        __syncthreads();
    }

    // store
    for (int ii = 0; ii < TILEM; ii += blockLayoutRowC)
    {
        for (int jj = 0; jj < TILEN; jj += blockLayoutColC)
        {
            if (row0 + ii + rowIdC < m && col0 + jj + colIdC < n)
            {
                C[batchStartC + (row0 + ii + rowIdC) * n + (col0 + jj + colIdC)] =
                    regSum[ii / blockLayoutRowC][jj / blockLayoutColC];
            }
        }
    }

    return;
}

int bmm_v5(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32;
    const int blockSize = 256;
    dim3 blockDims(blockSize);

    const int tileM = 128;
    const int tileN = 128;
    const int tileK = 32;
    // block 在处理 sharedTileA 时的布局;
    const int blockLayoutRowA = blockSize / (tileK / 4);   // 32
    const int blockLayoutColA = tileK / 4;                 // 8
    const int warpLayoutRowA = warpSize / blockLayoutColA; // 4
    const int warpLayoutColA = blockLayoutColA;            // 8
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

    bmm_kernel_v5<warpSize, blockSize, tileM, tileN, tileK,
                  blockLayoutRowA, blockLayoutColA, blockLayoutRowB, blockLayoutColB, blockLayoutRowC, blockLayoutColC,
                  warpLayoutRowA, warpLayoutColA, warpLayoutRowB, warpLayoutColB, warpLayoutRowC, warpLayoutColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}
