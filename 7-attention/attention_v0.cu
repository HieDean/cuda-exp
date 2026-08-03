#include "attention_vx.h"

template <int tileDimRow, int tileDimCol, int warpSize>
__global__ void attention_kernel_v0(const float *Q, const float *K, const float *V,
                                    const int *attnMasks, const float scale,
                                    float *L, float *M, float *O,
                                    int bs, int numHeads, int seqLen, int headDim,
                                    cudaStream_t stream)
{
    int numTilesRow = (seqLen + tileDimRow - 1) / tileDimRow;
    int numTilesCol = (seqLen + tileDimCol - 1) / tileDimCol;

    int batchStart = blockIdx.y * seqLen;
    int row0 = blockIdx.x * tileDimRow;

    int row = threadIdx.x / tileDimCol;
    int col = threadIdx.x % tileDimCol;

    float globalMax = -INFINITY;
    float globalSum = 0.0f;

    for (int tc = 0; tc < numTilesCol; ++tc)
    {
        int col0 = tc * tileDimCol;

        // tileQ @ tileK^T within one block
        float sum = 0.0f;
        for (int i = 0; i < headDim; ++i)
        {
            sum += row0 + row < seqLen && col0 + col < seqLen ? Q[(batchStart + row0 + row) * headDim + i] * K[(batchStart + col0 + col) * headDim + i] : 0.0f;
        }
        float weight = sum * scale;

        // online softmax
        // max and sum reduce within one warp
        float localMax = row0 + row < seqLen && col0 + col < seqLen ? weight : -INFINITY;
        float localSum = row0 + row < seqLen && col0 + col < seqLen ? 1.0f : 0.0F;
        for (int mask = tileDimCol >> 1; mask > 0; mask >>= 1)
        {
            float anotherLocalMax = __shfl_xor_sync(0xffffffff, localMax, mask, tileDimCol);
            float anotherlocalSum = __shfl_xor_sync(0xffffffff, localSum, mask, tileDimCol);

            float newLocalMax = fmaxf(localMax, anotherLocalMax);
            localSum = localSum * expf(localMax - newLocalMax) + anotherlocalSum * expf(anotherLocalMax - newLocalMax);
            localMax = newLocalMax;
        }

        // update global
        float newGlobalMax = fmaxf(localMax, globalMax);
        float newGlobalSum = globalSum * expf(globalMax - newGlobalMax) + localSum * expf(localMax - newGlobalMax);

        // div
        weight = expf(weight - newGlobalMax) / newGlobalSum;

        // attnWeights @ V
        for (int i = 0; i < headDim; ++i)
        {
            // sum reduce within one warp
            float weighted = col0 + col < seqLen ? weight * V[(batchStart + col0 + col) * headDim + i] : 0.0f;
            for (int mask = tileDimCol >> 1; mask > 0; mask >>= 1)
            {
                float anotherWeighted = __shfl_down_sync(0xffffffff, weighted, mask, tileDimCol);
                weighted += anotherWeighted;
            }
            if (row0 + row < seqLen && col0 + col < seqLen && col == 0)
            {
                O[(batchStart + row0 + row) * headDim + i] = weighted +
                                                O[(batchStart + row0 + row) * headDim + i] *
                                                    expf(globalMax - newGlobalMax) * globalSum / newGlobalSum;
            }
        }
        globalMax = newGlobalMax;
        globalSum = newGlobalSum;
    }

    if (row0 + row < seqLen && col == 0)
    {
        M[batchStart + row0 + row] = globalMax;
        L[batchStart + row0 + row] = globalSum;
    }

    return;
}

int attention_v0(const float *Q, const float *K, const float *V,
                 const int *attnMasks,
                 float *L, float *M, float *O,
                 int bs, int numHeads, int seqLen, int headDim,
                 cudaStream_t stream)
{
    const int warpSize = 32;
    const int blockSize = 256;
    dim3 blockDims(blockSize);

    const int tileDimRow = blockSize / warpSize; // 8
    const int tileDimCol = warpSize;             // 32
    int numTilesRow = (seqLen + tileDimRow - 1) / tileDimRow;
    int numTilesCol = (seqLen + tileDimCol - 1) / tileDimCol;

    int gridDim_y = bs * numHeads;
    int gridDim_x = numTilesRow;
    dim3 gridDims(gridDim_x, gridDim_y);

    float inv_sqrt_d = rsqrtf(static_cast<float>(headDim));
    attention_kernel_v0<tileDimRow, tileDimCol, warpSize>
        <<<gridDims, blockDims, 0, stream>>>(Q, K, V, attnMasks, inv_sqrt_d,
                                             L, M, O,
                                             bs, numHeads, seqLen, headDim, stream);

    return 0;
}