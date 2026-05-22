#include <cuda_runtime.h>
#include <cublas_v2.h>

int transpose_v0(const float *A, float *B,
                 int m, int n, cudaStream_t stream);

int transpose_v1(const float *A, float *B,
                 int m, int n, cudaStream_t stream);
