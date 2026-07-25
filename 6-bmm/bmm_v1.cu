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

        for (int tk = 0; tk < TILEK; ++tk)
        {
            for (int ii = 0; ii < TILEM; ii += BLOCKROWC)
            {
                for (int jj = 0; jj < TILEN; jj += BLOCKCOLC)
                {
                    regSum[ii / BLOCKROWC][jj / BLOCKCOLC] +=
                        sharedTileA[ii + rowIdC][tk] * sharedTileB[tk][jj + colIdC];
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

/**
 * tileM = tileN = 32 64 128 256
 * tileK = 4 8 16 32 64
 * 实测下来, 性能最好的是 bmm_v1_128_8;
 * 但实际上, 在不同的输入 shape 条件下, 性能最优的 kernel 是不一样的, 所以不能一概而论;
 */
int bmm_v1_32_8(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 32; const int tileN = 32; const int tileK = 8;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_32_16(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 32; const int tileN = 32; const int tileK = 16;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_32_32(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 32; const int tileN = 32; const int tileK = 32;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_32_64(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 32; const int tileN = 32; const int tileK = 64;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_64_4(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 64; const int tileN = 64; const int tileK = 4;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_64_8(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 64; const int tileN = 64; const int tileK = 8;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_64_16(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 64; const int tileN = 64; const int tileK = 16;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_64_32(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 64; const int tileN = 64; const int tileK = 32;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_64_64(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 64; const int tileN = 64; const int tileK = 64;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_128_4(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 128; const int tileN = 128; const int tileK = 4;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_128_8(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 128; const int tileN = 128; const int tileK = 8;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_128_16(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 128; const int tileN = 128; const int tileK = 16;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_128_32(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 128; const int tileN = 128; const int tileK = 32;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_256_4(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 256; const int tileN = 256; const int tileK = 4;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_256_8(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 256; const int tileN = 256; const int tileK = 8;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}

int bmm_v1_256_16(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32; const int blockSize = 256; dim3 blockDims(blockSize);

    const int tileM = 256; const int tileN = 256; const int tileK = 16;
    // block 在处理 sharedTileA 时的布局;
    const int blockRowA = blockSize / tileK; const int blockColA = tileK;
    // block 在处理 sharedTileB 时的布局;
    const int blockRowB = blockSize / tileN; const int blockColB = tileN;
    // block 在处理 TileC 时的布局;
    const int blockRowC = 16; const int blockColC = 16;

    int gridDim_z = bs; int gridDim_y = (m + tileM - 1) / tileM; int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v1<warpSize, blockSize, tileM, tileN, tileK,
                  blockRowA, blockColA, blockRowB, blockColB, blockRowC, blockColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}
