#include "bmm_vx.h"

template <int WARPSIZE, int BLOCKSIZE,
          int TILEM, int TILEN, int TILEK,
          int blockLayoutRowA, int blockLayoutColA,
          int blockLayoutRowB, int blockLayoutColB,
          int blockLayoutRowC, int blockLayoutColC,
          int warpLayoutRowC, int warpLayoutColC>
__global__ void bmm_kernel_v3(const float *A, const float *B, float *C,
                              int bs, int m, int n, int k)
{
    int batchStartA = blockIdx.z * m * k;
    int batchStartB = blockIdx.z * n * k;
    int batchStartC = blockIdx.z * m * n;

    int row0 = blockIdx.y * TILEM;
    int col0 = blockIdx.x * TILEN;

    __shared__ float sharedTileA[TILEM][TILEK];
    __shared__ float sharedTileB[TILEK][TILEN];

    float regSum[TILEM / blockLayoutRowC][TILEN / blockLayoutColC] = {0.0f};
    float regA[TILEM / blockLayoutRowC] = {0.0f};
    float regB[TILEN / blockLayoutColC] = {0.0f};

    int tId = threadIdx.x;
    int rowIdA = tId / blockLayoutColA;
    int colIdA = tId % blockLayoutColA;
    int rowIdB = tId / blockLayoutColB;
    int colIdB = tId % blockLayoutColB;

    // warp 重组
    int warpId = tId / WARPSIZE;
    int numWarpsColC = blockLayoutColC / warpLayoutColC;
    [[maybe_unused]] int numWarpsRowC = blockLayoutRowC / warpLayoutRowC;
    int warpIdRowC = warpId / numWarpsColC;
    int warpIdColC = warpId % numWarpsColC;
    int laneId = tId % WARPSIZE;
    int laneIdRowC = laneId / warpLayoutColC;
    int laneIdColC = laneId % warpLayoutColC;
    // 将 warp layout 映射至 block layout
    int rowIdC = warpIdRowC * warpLayoutRowC + laneIdRowC;
    int colIdC = warpIdColC * warpLayoutColC + laneIdColC;

    // k loop
    for (int kk = 0; kk < k; kk += TILEK)
    {
        for (int ii = 0; ii < TILEM; ii += blockLayoutRowA)
        {
            for (int jj = 0; jj < TILEK; jj += blockLayoutColA)
            {
                sharedTileA[ii + rowIdA][jj + colIdA] =
                    row0 + ii + rowIdA < m && kk + jj + colIdA < k ?
                        A[batchStartA + (row0 + ii + rowIdA) * k + (kk + jj + colIdA)] : 0.0f;
            }
        }
        for (int ii = 0; ii < TILEK; ii += blockLayoutRowB)
        {
            for (int jj = 0; jj < TILEN; jj += blockLayoutColB)
            {
                sharedTileB[ii + rowIdB][jj + colIdB] =
                    kk + ii + rowIdB < k && col0 + jj + colIdB < n ?
                        B[batchStartB + (kk + ii + rowIdB) * n + (col0 + jj + colIdB)] : 0.0f;
            }
        }
        __syncthreads();

        for (int tk = 0; tk < TILEK; ++tk)
        {
            for (int ii = 0; ii < TILEM; ii += blockLayoutRowC) {
                regA[ii / blockLayoutRowC] = sharedTileA[ii + rowIdC][tk];
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

int bmm_v3_2_16(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32;
    const int blockSize = 256;
    dim3 blockDims(blockSize);

    const int tileM = 128;
    const int tileN = 128;
    const int tileK = 32;
    // block 在处理 sharedTileA 时的布局;
    const int blockLayoutRowA = blockSize / tileK;
    const int blockLayoutColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockLayoutRowB = blockSize / tileN;
    const int blockLayoutColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockLayoutRowC = 16;
    const int blockLayoutColC = 16;
    // warpLayoutRowC <= blockLayoutRowC
    // warpLayoutColC <= blockLayoutColC
    // warpLayoutRowC * warpLayoutColC = warpSize
    const int warpLayoutRowC = 2;
    const int warpLayoutColC = 16;

    int gridDim_z = bs;
    int gridDim_y = (m + tileM - 1) / tileM;
    int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v3<warpSize, blockSize, tileM, tileN, tileK,
                  blockLayoutRowA, blockLayoutColA, blockLayoutRowB, blockLayoutColB, blockLayoutRowC, blockLayoutColC,
                  warpLayoutRowC, warpLayoutColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v3_4_8(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32;
    const int blockSize = 256;
    dim3 blockDims(blockSize);

    const int tileM = 128;
    const int tileN = 128;
    const int tileK = 32;
    // block 在处理 sharedTileA 时的布局;
    const int blockLayoutRowA = blockSize / tileK;
    const int blockLayoutColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockLayoutRowB = blockSize / tileN;
    const int blockLayoutColB = tileN;
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

    bmm_kernel_v3<warpSize, blockSize, tileM, tileN, tileK,
                  blockLayoutRowA, blockLayoutColA, blockLayoutRowB, blockLayoutColB, blockLayoutRowC, blockLayoutColC,
                  warpLayoutRowC, warpLayoutColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

// int bmm_v3_8_4(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
// {
//     const int warpSize = 32;
//     const int blockSize = 256;
//     dim3 blockDims(blockSize);

//     const int tileM = 128;
//     const int tileN = 128;
//     const int tileK = 32;
//     // block 在处理 sharedTileA 时的布局;
//     const int blockLayoutRowA = blockSize / tileK;
//     const int blockLayoutColA = tileK;
//     // block 在处理 sharedTileB 时的布局;
//     const int blockLayoutRowB = blockSize / tileN;
//     const int blockLayoutColB = tileN;
//     // block 在处理 TileC 时的布局;
//     const int blockLayoutRowC = 16;
//     const int blockLayoutColC = 16;
//     // warpLayoutRowC <= blockLayoutRowC
//     // warpLayoutColC <= blockLayoutColC
//     // warpLayoutRowC * warpLayoutColC = warpSize
//     const int warpLayoutRowC = 8;
//     const int warpLayoutColC = 4;

//     int gridDim_z = bs;
//     int gridDim_y = (m + tileM - 1) / tileM;
//     int gridDim_x = (n + tileN - 1) / tileN;
//     dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

//     bmm_kernel_v3<warpSize, blockSize, tileM, tileN, tileK,
//                   blockLayoutRowA, blockLayoutColA, blockLayoutRowB, blockLayoutColB, blockLayoutRowC, blockLayoutColC,
//                   warpLayoutRowC, warpLayoutColC>
//         <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
//     return 0;
// }

// int bmm_v3_16_2(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
// {
//     const int warpSize = 32;
//     const int blockSize = 256;
//     dim3 blockDims(blockSize);

//     const int tileM = 128;
//     const int tileN = 128;
//     const int tileK = 32;
//     // block 在处理 sharedTileA 时的布局;
//     const int blockLayoutRowA = blockSize / tileK;
//     const int blockLayoutColA = tileK;
//     // block 在处理 sharedTileB 时的布局;
//     const int blockLayoutRowB = blockSize / tileN;
//     const int blockLayoutColB = tileN;
//     // block 在处理 TileC 时的布局;
//     const int blockLayoutRowC = 16;
//     const int blockLayoutColC = 16;
//     // warpLayoutRowC <= blockLayoutRowC
//     // warpLayoutColC <= blockLayoutColC
//     // warpLayoutRowC * warpLayoutColC = warpSize
//     const int warpLayoutRowC = 16;
//     const int warpLayoutColC = 2;

//     int gridDim_z = bs;
//     int gridDim_y = (m + tileM - 1) / tileM;
//     int gridDim_x = (n + tileN - 1) / tileN;
//     dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

//     bmm_kernel_v3<warpSize, blockSize, tileM, tileN, tileK,
//                   blockLayoutRowA, blockLayoutColA, blockLayoutRowB, blockLayoutColB, blockLayoutRowC, blockLayoutColC,
//                   warpLayoutRowC, warpLayoutColC>
//         <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
//     return 0;
// }
