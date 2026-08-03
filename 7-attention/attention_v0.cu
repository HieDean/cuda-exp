#include "attention_vx.h"

template <int tileDimRow, int tileDimCol, int numTilesRow, int numTilesCol, int warpSize>
__global__ void attention_kernel_v0(const float *Q, const float *K, const float *V,
                                    const float *attnMasks, const float scale, 
                                    float *L, float *M, float *O,
                                    int bs, int numHeads, int seqLen, int headDim,
                                    cudaStream_t stream)
{
    int batchStart = blockIdx.y * seqLen * headDim;
    int row0 = batchStart + blockIdx.x * tileDimRow * headDim;

    int row = threadIdx.x / tileDimCol;
    int col = threadIdx.x % tileDimCol;

    __shared__ float sharedAttnWeights[tileDimRow][tileDimCol];
    __shared__ float sharedMax[tileDimRow] = {-INFINITY};
    __shared__ float sharedSum[tileDimRow] = {0.0f};

    for (int tc = 0; tc < numTilesCol; ++tc)
    {
        int col0 = batchStart + tc * tileDimCol * headDim;

        // tileQ @ tileK^T within one block
        float sum = 0.0f;
        for (int i = 0; i < headDim; ++i)
        {
            sum += Q[row0 + row * headDim + i] * K[col0 + col * headDim + i];
        }
        weight = sum * scale;

        // online softmax
        // max and sum reduce within one warp
        float localSum, localMax;
        localSum = localMax = weight;
        for (int mask = tileDimCol >> 1; mask > 0; mask >>= 1)
        {
            float anotherLocalMax = __shfl_xor_sync(0xffffffff, localMax, mask, tileDimCol);
            float anotherlocalSum = __shfl_xor_sync(0xffffffff, localSum, mask, tileDimCol);

            float newLocalMax = fmaxf(localMax, anotherLocalMax);
            localSum = localSum * expf(localMax - newLocalMax) + anotherlocalSum * expf(anotherLocalMax - newLocalMax);
            localMax = newLocalMax;
        }

        float m = M[blockIdx.y * seqLen + row];
        float l = L[blockIdx.y * seqLen + row];

        float globalMax = fmaxf(localMax, m);
        float globalSum = l * expf(m - globalMax) + localSum * expf(localMax - globalMax);

        // div
        weight = expf(weight - globalMax) / globalSum;

        // attnWeights @ V
        for (int i = 0; i < headDim; ++i)
        {
            // sum reduce within one warp
            float weighted = weight * V[col0 + col * headDim + i];
            for (int mask = tileDimCol >> 1; mask > 0; mask >>= 1)
            {
                float anotherWeighted = __shfl_down_sync(0xffffffff, weighted, mask, tileDimCol);
                weighted += anotherWeighted;
            }
            if (col == 0)
            {
                O[row0 + row * headDim + i] = weighted + O[row0 + row * headDim + i] * expf(m - globalMax) * l / globalSum;
                M[blockIdx.y * seqLen + row] = globalMax;
                L[blockIdx.y * seqLen + row] = globalSum;
            }
        }
    }

    return;
}

int attention_v0(const float *Q, const float *K, const float *V,
                 const float *attnMasks,
                 float *L, float *M, float *O,
                 int bs, int numHeads, int seqLen, int headDim,
                 cudaStream_t stream);
{
    const int warpSize = 32;
    const int blockSize = 256;
    dim3 blockDims(blockSize);

    const int tileDimRow = blockSize / warpSize; // 8
    const int tileDimCol = warpSize;             // 32
    const int numTilesRow = (seqLen + tileDimRow - 1) / tileDimRow;
    const int numTilesCol = (seqLen + tileDimCol - 1) / tileDimCol;

    int gridDim_y = bs * numHeads;
    int gridDim_x = numTilesRow;
    dim3 gridDims(gridDim_x, gridDim_y, gridDim_z);

    float inv_sqrt_d = rsqrtf(static_cast<float>(head_dim));
    attention_kernel_v0<tileDimRow, tileDimCol, numTilesRow, numTilesCol, warpSize>
        <<<gridDims, blockDims, 0, stream>>>(Q, K, V, attnMasks, inv_sqrt_d,
                                             L, M, O,
                                             bs, numHeads, seqLen, headDim, stream);

    return 0;
}