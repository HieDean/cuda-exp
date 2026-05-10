#include <cuda_runtime.h>
#include <cublas_v2.h>

int dot_v0(const float *A, const float *B, float *C,
            int length, cudaStream_t stream);

// int dot_v1(const float *A, const float *B, float *C,
//             int m, int n, int k, cudaStream_t stream);

// int dot_v2(const float *A, const float *B, float *C,
//             int m, int n, int k, cudaStream_t stream);

// int dot_v3(const float *A, const float *B, float *C,
//             int m, int n, int k, cudaStream_t stream);

// int dot_v4(const float *A, const float *B, float *C,
//             int m, int n, int k, cudaStream_t stream);
