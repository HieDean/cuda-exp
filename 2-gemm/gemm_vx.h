#include <cuda_runtime.h>
#include <cublas_v2.h>

int gemm_v0(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream);

int gemm_v1(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream);

int gemm_v2(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream);

int gemm_v3(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream);
