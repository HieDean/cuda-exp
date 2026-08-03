#include <cuda_runtime.h>

int attention_v0(const float *Q, const float *K, const float *V,
                 const float *attnMasks,
                 float *L, float *M, float *O,
                 int bs, int numHeads, int seqLen, int headDim,
                 cudaStream_t stream);
