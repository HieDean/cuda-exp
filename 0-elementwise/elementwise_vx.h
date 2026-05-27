#include <cuda_runtime.h>
#include <cublas_v2.h>

int elementwise_v0(const float *A, const float *B, float *C,
                 int m, int n, cudaStream_t stream);

// int elementwise_v1(const float *A, const float *B, float *C,
//                  int m, int n, cudaStream_t stream);

// int elementwise_v2(const float *A, const float *B, float *C,
//                  int m, int n, cudaStream_t stream);
