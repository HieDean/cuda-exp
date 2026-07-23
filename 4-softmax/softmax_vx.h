#include <cuda_runtime.h>
#include <cublas_v2.h>

int softmax_v0(const float *A, float *B,
               int n, cudaStream_t stream);
