#include <cuda_runtime.h>
#include <cublas_v2.h>

int bmm_v0(const float *A, const float *B, float *C,
           int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_32_8(const float *A, const float *B, float *C,
                int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_32_16(const float *A, const float *B, float *C,
                 int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_32_32(const float *A, const float *B, float *C,
                 int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_32_64(const float *A, const float *B, float *C,
                 int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_64_4(const float *A, const float *B, float *C,
                int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_64_8(const float *A, const float *B, float *C,
                int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_64_16(const float *A, const float *B, float *C,
                 int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_64_32(const float *A, const float *B, float *C,
                 int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_64_64(const float *A, const float *B, float *C,
                 int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_128_4(const float *A, const float *B, float *C,
                 int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_128_8(const float *A, const float *B, float *C,
                 int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_128_16(const float *A, const float *B, float *C,
                  int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_128_32(const float *A, const float *B, float *C,
                  int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_256_4(const float *A, const float *B, float *C,
                 int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_256_8(const float *A, const float *B, float *C,
                 int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v1_256_16(const float *A, const float *B, float *C,
                  int bs, int m, int n, int k, cudaStream_t stream);

int bmm_v2(const float *A, const float *B, float *C,
           int bs, int m, int n, int k, cudaStream_t stream);