#include "bmm_vx.h"

#include <cassert>
#include <mma.h>
using namespace nvcuda;

template <int warpSize, int blockSize,
          int tileM, int tileN, int tileK,
          int wmmaM, int wmmaN, int wmmaK,
          int blockLayoutRowA, int blockLayoutColA,
          int blockLayoutRowB, int blockLayoutColB,
          int blockLayoutRowC, int blockLayoutColC,
          int warpLayoutRowC, int warpLayoutColC>
__global__ void bmm_kernel_v7(const float *A, const float *B, float *C,
                              int bs, int m, int n, int k)
{
    int batchStartA = blockIdx.z * m * k;
    int batchStartB = blockIdx.z * n * k;
    int batchStartC = blockIdx.z * m * n;

    int row0 = blockIdx.y * tileM;
    int col0 = blockIdx.x * tileN;

    const int numFragsRowC = tileM / wmmaM;
    const int numFragsColC = tileN / wmmaN;

    __shared__ half sharedFragsA[numFragsRowC][wmmaM * wmmaK];
    __shared__ half sharedFragsB[numFragsColC][wmmaK * wmmaN];
    __shared__ float sharedFragsC[numFragsColC][wmmaM * wmmaN];

    wmma::fragment<wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, wmmaM, wmmaN, wmmaK, float> c_frag[numFragsRowC];
    for (int ii = 0; ii < numFragsRowC; ++ii)
    {
        wmma::fill_fragment(c_frag[ii], 0.0f);
    }

    int tId = threadIdx.x;
    int warpId = tId / warpSize;
    int laneId = tId % warpSize;

    int rowIdA = tId / blockLayoutColA;
    int colIdA = tId % blockLayoutColA;
    int rowIdB = tId / blockLayoutColB;
    int colIdB = tId % blockLayoutColB;

    [[maybe_unused]] int numWarpsRowC = blockLayoutRowC / warpLayoutRowC;
    int numWarpsColC = blockLayoutColC / warpLayoutColC;
    int warpIdRowC = warpId / numWarpsColC;
    int warpIdColC = warpId % numWarpsColC;
    int laneIdRowC = laneId / warpLayoutColC;
    int laneIdColC = laneId % warpLayoutColC;
    int rowIdC = warpIdRowC * warpLayoutRowC + laneIdRowC;
    int colIdC = warpIdColC * warpLayoutColC + laneIdColC;

    // k loop
    for (int kk = 0; kk < k; kk += tileK)
    {
        for (int ii = 0; ii < tileM; ii += blockLayoutRowA)
        {
            for (int jj = 0; jj < tileK; jj += blockLayoutColA)
            {
                sharedFragsA[(ii + rowIdA) / wmmaM][(ii + rowIdA) % wmmaM * wmmaK + jj + colIdA] =
                    row0 + ii + rowIdA < m && kk + jj + colIdA < k ? __float2half(A[batchStartA + (row0 + ii + rowIdA) * k + (kk + jj + colIdA)]) : __float2half(0.0f);
            }
        }
        for (int ii = 0; ii < tileK; ii += blockLayoutRowB)
        {
            for (int jj = 0; jj < tileN; jj += blockLayoutColB)
            {
                sharedFragsB[(jj + colIdB) / wmmaN][(ii + rowIdB) * wmmaN + (jj + colIdB) % wmmaN] =
                    kk + ii + rowIdB < k && col0 + jj + colIdB < n ? __float2half(B[batchStartB + (kk + ii + rowIdB) * n + (col0 + jj + colIdB)]) : __float2half(0.0f);
            }
        }
        __syncthreads();

        wmma::load_matrix_sync(b_frag, sharedFragsB[warpId], wmmaN);
        for (int ii = 0; ii < tileM; ii += wmmaM)
        {
            wmma::load_matrix_sync(a_frag, sharedFragsA[ii / wmmaM], wmmaK);
            wmma::mma_sync(c_frag[ii / wmmaM], a_frag, b_frag, c_frag[ii / wmmaM]);
        }
        __syncthreads();
    }

    // store
    for (int ii = 0; ii < tileM; ii += blockLayoutRowC)
    {
        wmma::store_matrix_sync(sharedFragsC[warpId], c_frag[ii / wmmaM], wmmaN, wmma::mem_row_major);
        __syncthreads();
        for (int jj = 0; jj < tileN; jj += blockLayoutColC)
        {
            if (row0 + ii + rowIdC < m && col0 + jj + colIdC < n)
            {
                C[batchStartC + (row0 + ii + rowIdC) * n + (col0 + jj + colIdC)] =
                    sharedFragsC[(jj + colIdC) / wmmaN][(ii + rowIdC) % wmmaM * wmmaN + (jj + colIdC) % wmmaN];
            }
        }
        __syncthreads();
    }

    return;
}

int bmm_v7(const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream)
{
    const int warpSize = 32;
    const int blockSize = 256;
    dim3 blockDims(blockSize);

    const int wmmaM = 16;
    const int wmmaN = 16;
    const int wmmaK = 16;
    const int tileM = 128;
    const int tileN = 128;
    const int tileK = 16;
    // block 在处理 sharedTileA 时的布局;
    const int blockLayoutRowA = blockSize / tileK; // 16
    const int blockLayoutColA = tileK;             // 16
    // block 在处理 sharedTileB 时的布局;
    const int blockLayoutRowB = blockSize / tileN; // 2
    const int blockLayoutColB = tileN;             // 128
    // block 在处理 TileC 时的布局;
    const int blockLayoutRowC = 16;
    const int blockLayoutColC = 16;
    const int warpLayoutRowC = 4;
    const int warpLayoutColC = 8;

    assert (tileK == wmmaK);
    assert (blockSize / warpSize == tileN / wmmaN);
    assert (blockLayoutRowC == wmmaM);

    int gridDim_z = bs;
    int gridDim_y = (m + tileM - 1) / tileM;
    int gridDim_x = (n + tileN - 1) / tileN;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    bmm_kernel_v7<warpSize, blockSize, tileM, tileN, tileK, wmmaM, wmmaN, wmmaK,
                  blockLayoutRowA, blockLayoutColA, blockLayoutRowB, blockLayoutColB, blockLayoutRowC, blockLayoutColC,
                  warpLayoutRowC, warpLayoutColC>
        <<<gridDims, blockDims, 0, stream>>>(A, B, C, bs, m, n, k);
    return 0;
}
