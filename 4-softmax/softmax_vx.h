#include <cuda_runtime.h>
#include <cublas_v2.h>

int softmax_v0(const float *A, float *B,
               int bs, int num, cudaStream_t stream);

int softmax_v1(const float *A, float *B,
               int bs, int num, cudaStream_t stream);
