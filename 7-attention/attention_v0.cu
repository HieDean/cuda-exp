#include "attention_vx.h"

template <int TILEM, int TILEN, int TILEK, int BLOCKSIZE, int WARPSIZE>
__global__ void attention_kernel_v0(const float *A, const float *B, float *C,
                              int bs, int m, int n, int k)
{

    return;
}

int attention_v0(const float *Q, const float *K, const float *V,
                 const float *attnMasks,
                 float *attnWeights, float *O,
                 int bs, int numHeads, int seqLen, int headDim,
                 cudaStream_t stream);
{
    return 0;
}