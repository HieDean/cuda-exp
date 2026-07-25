#include "bmm_vx.h"

template <int TILEM, int TILEN, int TILEK, int BLOCKSIZE, int WARPSIZE>
__global__ void bmm_kernel_v0(const float *A, const float *B, float *C,
                              int bs, int m, int n, int k)
{
    int batchStartA = blockIdx.z * m * k;
    int batchStartB = blockIdx.z * n * k;
    int batchStartC = blockIdx.z * m * n;

    int row0 = blockIdx.y * TILEM;
    int col0 = blockIdx.x * TILEN;

    __shared__ float sharedTileA[TILEM][TILEK];
    __shared__ float sharedTileB[TILEK][TILEN];

    int tId = threadIdx.x;
    int rowIdA = tId / TILEK;
    int colIdA = tId % TILEK;
    int rowIdB = tId / TILEN;
    int colIdB = tId % TILEN;
    int rowIdC = tId / TILEN;
    int colIdC = tId % TILEN;

    // k loop
    float sum = 0.0;
    for (int kk = 0; kk < k; kk += TILEK)
    {
        if (rowIdA < TILEM && colIdA < TILEK)
        {
            sharedTileA[rowIdA][colIdA] = row0 + rowIdA < m && kk + colIdA < k ?
                A[batchStartA + (row0 + rowIdA) * k + (kk + colIdA)] : 0.0f;
        }
        if (rowIdB < TILEK && colIdB < TILEN)
        {
            sharedTileB[rowIdB][colIdB] = kk + rowIdB < k && col0 + colIdB < n ?
                B[batchStartB + (kk + rowIdB) * n + (col0 + colIdB)] : 0.0f;
        }
        __syncthreads();

        for (int tk = 0; tk < TILEK; ++tk)
        {
            sum += sharedTileA[rowIdC][tk] * sharedTileB[tk][colIdC];
        }
        __syncthreads();
    }

    if (row0 + rowIdC < m && col0 + colIdC < n)
    {
        C[batchStartC + (row0 + rowIdC) * n + col0 + colIdC] = sum;
    }

    return;
}

int bmm_v0(const float *A, const float *B, float *C,
           int bs, int m, int n, int k, cudaStream_t stream)
{
    const int tileM = 16;
    const int tileN = 16;
    const int tileK = 16;
    const int blockSize = 256; // tileM * tileN == blockSize
                               // tileM * tileK <= blockSize
                               // tileK * tileN <= blockSize
    const int warpSize = 32;

    dim3 blockDims(blockSize);

    int gridDim_z = bs;
    int gridDim_y = (m + tileM - 1) / tileM;
    int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v0<tileM, tileN, tileK, blockSize, warpSize>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}